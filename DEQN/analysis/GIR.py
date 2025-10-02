import jax
from jax import lax, random
from jax import numpy as jnp


def create_GIR_fn(econ_model, config, simul_policies=None):
    """
    Create a function to compute Generalized Impulse Responses (GIRs) for analysis variables.

    Args:
        econ_model: Economic model instance
        config: Dictionary with parameters:
            - gir_n_draws: Number of points to sample from ergodic distribution
            - gir_trajectory_length: Length of trajectories to simulate
            - shock_size: Size of shock to apply (default 0.2 for 20% decrease)
            - states_to_shock: List of state indices to shock (if None, shocks all available states)
            - gir_seed: Random seed
        simul_policies: Simulation policies for extracting price weights (optional, can be passed in GIR_fn)
                       Should be in logdev form when using simulation_verbose_fn

    Note: This function expects inputs from simulation_verbose_fn which returns states and policies
          in log deviation form (not normalized by standard deviations).
    """

    def random_draws(simul_obs, n_draws, seed=0):
        """Sample random points from the ergodic distribution"""
        n_simul = simul_obs.shape[0]
        key = random.PRNGKey(seed)
        indices = random.choice(key, n_simul, shape=(n_draws,), replace=False)
        obs_draws = simul_obs[indices, :]
        return obs_draws

    def extract_price_weights(simul_policies_data):
        """Extract price weights from simulation policies"""
        # Get average prices from simulation policies (following stochastic SS pattern)
        simul_policies_mean = jnp.mean(simul_policies_data, axis=0)
        P_weights = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
        Pk_weights = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
        Pm_weights = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]
        return P_weights, Pk_weights, Pm_weights

    def create_counterfactual_state(state_logdev, state_idx, shock_size):
        """
        Create counterfactual initial state with specified state variable shocked.

        Args:
            state_logdev: State in log deviation form (not normalized)
            state_idx: Index of state variable to shock
            shock_size: Size of negative shock to apply (positive value)

        Returns:
            counterfactual_state_logdev: Modified state in log deviation form
        """
        # Convert logdevs to actual log levels
        state_notnorm = state_logdev + econ_model.state_ss

        # Apply negative shock to specified state variable
        # Since state is in log terms, we subtract log(1 + shock_size) ≈ shock_size for small shocks
        state_counterfactual_notnorm = state_notnorm.at[state_idx].add(-jnp.log(1 + shock_size))

        # Convert back to logdevs
        state_counterfactual_logdev = state_counterfactual_notnorm - econ_model.state_ss

        return state_counterfactual_logdev

    def simul_trajectory_analysis_variables(
        econ_model, train_state, shocks, obs_init_logdev, P_weights, Pk_weights, Pm_weights, var_labels
    ):
        """
        Simulate full trajectory and return analysis variables at each step.

        Args:
            econ_model: Economic model
            train_state: Trained neural network state
            shocks: Sequence of shocks
            obs_init_logdev: Initial observation in log deviation form
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation
            var_labels: List of variable labels in consistent order

        Returns:
            trajectory_analysis_variables: Analysis variables for each time step [T+1, n_analysis_vars]
        """

        def step(obs_logdev, shock):
            # Convert logdevs to normalized state for neural network and econ_model calls
            obs_normalized = obs_logdev / econ_model.state_sd
            policy_normalized = train_state.apply_fn(train_state.params, obs_normalized)
            next_obs_normalized = econ_model.step(obs_normalized, policy_normalized, shock)

            # Convert back to logdevs for consistency
            next_obs_logdev = next_obs_normalized * econ_model.state_sd
            policy_logdev = policy_normalized * econ_model.policies_sd

            # Compute analysis variables and convert to array using consistent label order
            analysis_vars_dict = econ_model.get_analysis_variables(
                obs_logdev, policy_logdev, P_weights, Pk_weights, Pm_weights
            )
            analysis_vars_array = jnp.array([analysis_vars_dict[label] for label in var_labels])
            return next_obs_logdev, analysis_vars_array

        final_obs_logdev, trajectory_analysis_variables = lax.scan(step, obs_init_logdev, shocks)

        # Compute analysis variables for the final observation (using last policy)
        final_obs_normalized = final_obs_logdev / econ_model.state_sd
        final_policy_normalized = train_state.apply_fn(train_state.params, final_obs_normalized)
        final_policy_logdev = final_policy_normalized * econ_model.policies_sd
        final_analysis_vars_dict = econ_model.get_analysis_variables(
            final_obs_logdev, final_policy_logdev, P_weights, Pk_weights, Pm_weights
        )
        final_analysis_vars_array = jnp.array([final_analysis_vars_dict[label] for label in var_labels])

        # Add final analysis variables to trajectory
        trajectory_analysis_variables_full = jnp.concatenate(
            [trajectory_analysis_variables, final_analysis_vars_array[None, :]], axis=0
        )

        return trajectory_analysis_variables_full

    def generate_shocks_for_T(key, T, n_sectors):
        """Generate random shock trajectory for T periods"""
        keys_t = random.split(key, T)
        return jax.vmap(lambda k: econ_model.sample_shock(k))(keys_t)  # shape [T, n_sectors]

    def compute_state_GIR(simul_obs, train_state, state_idx, P_weights, Pk_weights, Pm_weights, var_labels):
        """
        Compute GIR for a specific state variable using analysis variables.

        Args:
            simul_obs: Simulation observations from ergodic distribution (in logdev form)
            train_state: Trained neural network state
            state_idx: Index of state to shock
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation
            var_labels: List of variable labels in consistent order

        Returns:
            gir_analysis_variables: Averaged impulse response for analysis variables [T+1, n_analysis_vars]
        """
        # Sample points from ergodic distribution
        sample_points = random_draws(simul_obs, config["gir_n_draws"], config["gir_seed"])

        # Function to compute impulse response for a single initial point
        def single_point_IR(i, obs_init_logdev):
            # Generate same future shocks for both paths
            key_i = random.fold_in(random.PRNGKey(config["gir_seed"]), i)
            shocks = generate_shocks_for_T(key_i, config["gir_trajectory_length"], econ_model.n_sectors)

            # Original trajectory analysis variables
            traj_analysis_vars_orig = simul_trajectory_analysis_variables(
                econ_model, train_state, shocks, obs_init_logdev, P_weights, Pk_weights, Pm_weights, var_labels
            )

            # Create counterfactual initial state
            obs_counterfactual_logdev = create_counterfactual_state(
                obs_init_logdev, state_idx, config.get("shock_size", 0.2)
            )

            # Counterfactual trajectory analysis variables (using same shocks)
            traj_analysis_vars_counter = simul_trajectory_analysis_variables(
                econ_model,
                train_state,
                shocks,
                obs_counterfactual_logdev,
                P_weights,
                Pk_weights,
                Pm_weights,
                var_labels,
            )

            # Impulse response is difference
            ir_analysis_variables = traj_analysis_vars_counter - traj_analysis_vars_orig

            return ir_analysis_variables

        # Vectorize over all sample points with their indices
        idxs = jnp.arange(config["gir_n_draws"])
        all_ir_analysis_variables = jax.vmap(single_point_IR)(idxs, sample_points)

        # Average across sample points
        gir_analysis_variables = jnp.mean(all_ir_analysis_variables, axis=0)

        return gir_analysis_variables

    def GIR_fn(simul_obs, train_state, simul_policies_data=None):
        """
        Main GIR function that computes analysis variable impulse responses for specified states.

        Args:
            simul_obs: Simulation observations from ergodic distribution (in logdev form)
            train_state: Trained neural network state
            simul_policies_data: Simulation policies for extracting price weights (in logdev form)

        Returns:
            gir_results: Dictionary with analysis variable impulse responses for each state
        """
        # Use provided simul_policies or the one passed during creation
        if simul_policies_data is None:
            simul_policies_data = simul_policies

        if simul_policies_data is None:
            raise ValueError("simul_policies must be provided either during create_GIR_fn or GIR_fn call")

        # Extract price weights from simulation policies
        P_weights, Pk_weights, Pm_weights = extract_price_weights(simul_policies_data)

        # Get variable labels from a single analysis call
        first_analysis_vars = econ_model.get_analysis_variables(
            simul_obs[0], simul_policies_data[0], P_weights, Pk_weights, Pm_weights
        )
        var_labels = list(first_analysis_vars.keys())

        # Determine which states to shock
        states_to_shock = config.get("states_to_shock", None)
        if states_to_shock is None:
            # Default to all states in the model
            states_to_shock = list(range(simul_obs.shape[1]))

        gir_results = {}

        # Compute GIR for each specified state
        for state_idx in states_to_shock:
            gir_analysis_vars_array = compute_state_GIR(
                simul_obs, train_state, state_idx, P_weights, Pk_weights, Pm_weights, var_labels
            )

            # Convert array back to dictionary format
            gir_analysis_vars_dict = {label: gir_analysis_vars_array[:, i] for i, label in enumerate(var_labels)}

            # Create state label
            state_name = f"state_{state_idx}"
            if hasattr(econ_model, "state_labels") and state_idx < len(econ_model.state_labels):
                state_name = econ_model.state_labels[state_idx]

            gir_results[state_name] = {
                "gir_analysis_variables": gir_analysis_vars_dict,
                "state_idx": state_idx,
            }

        return gir_results

    return GIR_fn
