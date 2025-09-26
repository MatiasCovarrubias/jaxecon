import jax
from jax import lax, random
from jax import numpy as jnp


def create_GIR_fn(econ_model, config, simul_policies=None):
    """
    Create a function to compute Generalized Impulse Responses (GIRs) for aggregates.

    Args:
        econ_model: Economic model instance
        config: Dictionary with parameters:
            - gir_n_draws: Number of points to sample from ergodic distribution
            - gir_trajectory_length: Length of trajectories to simulate
            - gir_tfp_shock_size: Size of TFP shock (default 0.2 for 20% decrease)
            - gir_sectors_to_shock: List of sector indices to shock (if None, shocks all sectors)
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

    def create_counterfactual_state(state_logdev, sector_idx, shock_size):
        """
        Create counterfactual initial state with TFP decreased in specified sector.

        Args:
            state_logdev: State in log deviation form (not normalized)
            sector_idx: Index of sector to shock
            shock_size: Size of negative shock to apply (positive value)

        Returns:
            counterfactual_state_logdev: Modified state in log deviation form
        """
        # Convert logdevs to actual log levels
        state_notnorm = state_logdev + econ_model.state_ss

        # Extract TFP component (second half of state vector)
        a = state_notnorm[econ_model.n_sectors :].copy()

        # Apply negative shock to specified sector (decrease by shock_size)
        # Since a is in log terms, we subtract log(1 + shock_size) ≈ shock_size for small shocks
        a = a.at[sector_idx].add(-jnp.log(1 + shock_size))

        # Reconstruct state
        K_part = state_notnorm[: econ_model.n_sectors]
        state_counterfactual_notnorm = jnp.concatenate([K_part, a])

        # Convert back to logdevs
        state_counterfactual_logdev = state_counterfactual_notnorm - econ_model.state_ss

        return state_counterfactual_logdev

    def simul_trajectory_aggregates(
        econ_model, train_state, shocks, obs_init_logdev, P_weights, Pk_weights, Pm_weights
    ):
        """
        Simulate full trajectory and return aggregates at each step.

        Args:
            econ_model: Economic model
            train_state: Trained neural network state
            shocks: Sequence of shocks
            obs_init_logdev: Initial observation in log deviation form
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation

        Returns:
            trajectory_aggregates: Aggregates for each time step [T+1, n_aggregates]
        """

        def step(obs_logdev, shock):
            # Convert logdevs to normalized state for neural network and econ_model calls
            obs_normalized = obs_logdev / econ_model.state_sd
            policy_normalized = train_state.apply_fn(train_state.params, obs_normalized)
            next_obs_normalized = econ_model.step(obs_normalized, policy_normalized, shock)

            # Convert back to logdevs for consistency
            next_obs_logdev = next_obs_normalized * econ_model.state_sd
            policy_logdev = policy_normalized * econ_model.policies_sd

            # Compute aggregates using logdevs (get_aggregates expects logdevs as input)
            aggregates = econ_model.get_aggregates(obs_logdev, policy_logdev, P_weights, Pk_weights, Pm_weights)
            return next_obs_logdev, aggregates

        final_obs_logdev, trajectory_aggregates = lax.scan(step, obs_init_logdev, shocks)

        # Compute aggregates for the final observation (using last policy)
        final_obs_normalized = final_obs_logdev / econ_model.state_sd
        final_policy_normalized = train_state.apply_fn(train_state.params, final_obs_normalized)
        final_policy_logdev = final_policy_normalized * econ_model.policies_sd
        final_aggregates = econ_model.get_aggregates(
            final_obs_logdev, final_policy_logdev, P_weights, Pk_weights, Pm_weights
        )

        # Add final aggregates to trajectory
        trajectory_aggregates_full = jnp.concatenate([trajectory_aggregates, final_aggregates[None, :]], axis=0)

        return trajectory_aggregates_full

    def generate_shocks_for_T(key, T, n_sectors):
        """Generate random shock trajectory for T periods"""
        keys_t = random.split(key, T)
        return jax.vmap(lambda k: econ_model.sample_shock(k))(keys_t)  # shape [T, n_sectors]

    def compute_sector_GIR(simul_obs, train_state, sector_idx, P_weights, Pk_weights, Pm_weights):
        """
        Compute GIR for a specific sector using aggregates.

        Args:
            simul_obs: Simulation observations from ergodic distribution (in logdev form)
            train_state: Trained neural network state
            sector_idx: Index of sector to shock
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation

        Returns:
            gir_aggregates: Averaged impulse response for aggregates [T+1, n_aggregates]
        """
        # Sample points from ergodic distribution
        sample_points = random_draws(simul_obs, config["gir_n_draws"], config["gir_seed"])

        # Function to compute impulse response for a single initial point
        def single_point_IR(i, obs_init_logdev):
            # Generate same future shocks for both paths
            key_i = random.fold_in(random.PRNGKey(config["gir_seed"]), i)
            shocks = generate_shocks_for_T(key_i, config["gir_trajectory_length"], econ_model.n_sectors)

            # Original trajectory aggregates
            traj_agg_orig = simul_trajectory_aggregates(
                econ_model, train_state, shocks, obs_init_logdev, P_weights, Pk_weights, Pm_weights
            )

            # Create counterfactual initial state
            obs_counterfactual_logdev = create_counterfactual_state(
                obs_init_logdev, sector_idx, config.get("gir_tfp_shock_size", 0.2)
            )

            # Counterfactual trajectory aggregates (using same shocks)
            traj_agg_counter = simul_trajectory_aggregates(
                econ_model, train_state, shocks, obs_counterfactual_logdev, P_weights, Pk_weights, Pm_weights
            )

            # Impulse response is difference
            ir_aggregates = traj_agg_counter - traj_agg_orig

            return ir_aggregates

        # Vectorize over all sample points with their indices
        idxs = jnp.arange(config["gir_n_draws"])
        all_ir_aggregates = jax.vmap(single_point_IR)(idxs, sample_points)

        # Average across sample points
        gir_aggregates = jnp.mean(all_ir_aggregates, axis=0)

        return gir_aggregates

    def GIR_fn(simul_obs, train_state, simul_policies_data=None):
        """
        Main GIR function that computes aggregate impulse responses for specified sectors.

        Args:
            simul_obs: Simulation observations from ergodic distribution (in logdev form)
            train_state: Trained neural network state
            simul_policies_data: Simulation policies for extracting price weights (in logdev form)

        Returns:
            gir_results: Dictionary with aggregate impulse responses for each sector
        """
        # Use provided simul_policies or the one passed during creation
        if simul_policies_data is None:
            simul_policies_data = simul_policies

        if simul_policies_data is None:
            raise ValueError("simul_policies must be provided either during create_GIR_fn or GIR_fn call")

        # Extract price weights from simulation policies
        P_weights, Pk_weights, Pm_weights = extract_price_weights(simul_policies_data)

        # Determine which sectors to shock
        sectors_to_shock = config.get("gir_sectors_to_shock", None)
        if sectors_to_shock is None:
            sectors_to_shock = list(range(econ_model.n_sectors))

        gir_results = {}

        # Compute GIR for each specified sector
        for sector_idx in sectors_to_shock:
            gir_aggregates = compute_sector_GIR(simul_obs, train_state, sector_idx, P_weights, Pk_weights, Pm_weights)

            sector_name = f"sector_{sector_idx}"
            if hasattr(econ_model, "labels") and sector_idx < len(econ_model.labels):
                sector_name = econ_model.labels[sector_idx]

            gir_results[sector_name] = {
                "gir_aggregates": gir_aggregates,
                "sector_idx": sector_idx,
            }

        return gir_results

    return GIR_fn
