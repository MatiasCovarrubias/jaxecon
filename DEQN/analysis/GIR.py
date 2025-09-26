import jax
from jax import lax, random
from jax import numpy as jnp


def create_GIR_fn(econ_model, config, simul_policies=None):
    """
    Create a function to compute Generalized Impulse Responses (GIRs) for aggregates.

    Args:
        econ_model: Economic model instance
        config: Dictionary with parameters:
            - n_draws: Number of points to sample from ergodic distribution
            - trajectory_length: Length of trajectories to simulate
            - tfp_shock_size: Size of TFP shock (default 0.2 for 20% decrease)
            - sectors_to_shock: List of sector indices to shock (if None, shocks all sectors)
            - seed: Random seed
        simul_policies: Simulation policies for extracting price weights (optional, can be passed in GIR_fn)
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

    def create_counterfactual_state(state, sector_idx, shock_size):
        """
        Create counterfactual initial state with TFP decreased in specified sector.

        Args:
            state: Normalized state vector
            sector_idx: Index of sector to shock
            shock_size: Size of negative shock to apply (positive value)

        Returns:
            counterfactual_state: Modified normalized state
        """
        # Denormalize state
        state_notnorm = state * econ_model.state_sd + econ_model.state_ss

        # Extract TFP component (second half of state vector)
        a = state_notnorm[econ_model.n_sectors :].copy()

        # Apply negative shock to specified sector (decrease by shock_size)
        # Since a is in log terms, we subtract log(1 + shock_size) ≈ shock_size for small shocks
        a = a.at[sector_idx].add(-jnp.log(1 + shock_size))

        # Reconstruct state
        K_part = state_notnorm[: econ_model.n_sectors]
        state_counterfactual_notnorm = jnp.concatenate([K_part, a])

        # Renormalize
        state_counterfactual = (state_counterfactual_notnorm - econ_model.state_ss) / econ_model.state_sd

        return state_counterfactual

    def simul_trajectory_aggregates(econ_model, train_state, shocks, obs_init, P_weights, Pk_weights, Pm_weights):
        """
        Simulate full trajectory and return aggregates at each step.

        Args:
            econ_model: Economic model
            train_state: Trained neural network state
            shocks: Sequence of shocks
            obs_init: Initial observation
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation

        Returns:
            trajectory_aggregates: Aggregates for each time step [T+1, n_aggregates]
        """

        def step(obs, shock):
            policy = train_state.apply_fn(train_state.params, obs)
            next_obs = econ_model.step(obs, policy, shock)
            # Compute aggregates for current period
            aggregates = econ_model.get_aggregates(obs, policy, P_weights, Pk_weights, Pm_weights)
            return next_obs, aggregates

        final_obs, trajectory_aggregates = lax.scan(step, obs_init, shocks)

        # Compute aggregates for the final observation (using last policy)
        final_policy = train_state.apply_fn(train_state.params, final_obs)
        final_aggregates = econ_model.get_aggregates(final_obs, final_policy, P_weights, Pk_weights, Pm_weights)

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
            simul_obs: Simulation observations from ergodic distribution
            train_state: Trained neural network state
            sector_idx: Index of sector to shock
            P_weights, Pk_weights, Pm_weights: Price weights for aggregation

        Returns:
            gir_aggregates: Averaged impulse response for aggregates [T+1, n_aggregates]
        """
        # Sample points from ergodic distribution
        sample_points = random_draws(simul_obs, config["n_draws"], config["seed"])

        # Function to compute impulse response for a single initial point
        def single_point_IR(i, obs_init):
            # Generate same future shocks for both paths
            key_i = random.fold_in(random.PRNGKey(config["seed"]), i)
            shocks = generate_shocks_for_T(key_i, config["trajectory_length"], econ_model.n_sectors)

            # Original trajectory aggregates
            traj_agg_orig = simul_trajectory_aggregates(
                econ_model, train_state, shocks, obs_init, P_weights, Pk_weights, Pm_weights
            )

            # Create counterfactual initial state
            obs_counterfactual = create_counterfactual_state(obs_init, sector_idx, config.get("tfp_shock_size", 0.2))

            # Counterfactual trajectory aggregates (using same shocks)
            traj_agg_counter = simul_trajectory_aggregates(
                econ_model, train_state, shocks, obs_counterfactual, P_weights, Pk_weights, Pm_weights
            )

            # Impulse response is difference
            ir_aggregates = traj_agg_counter - traj_agg_orig

            return ir_aggregates

        # Vectorize over all sample points with their indices
        idxs = jnp.arange(config["n_draws"])
        all_ir_aggregates = jax.vmap(single_point_IR)(idxs, sample_points)

        # Average across sample points
        gir_aggregates = jnp.mean(all_ir_aggregates, axis=0)

        return gir_aggregates

    def GIR_fn(simul_obs, train_state, simul_policies_data=None):
        """
        Main GIR function that computes aggregate impulse responses for specified sectors.

        Args:
            simul_obs: Simulation observations from ergodic distribution
            train_state: Trained neural network state
            simul_policies_data: Simulation policies for extracting price weights

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
        sectors_to_shock = config.get("sectors_to_shock", None)
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
