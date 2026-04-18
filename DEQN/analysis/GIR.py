import jax
from jax import lax, random
from jax import numpy as jnp

from DEQN.analysis.model_hooks import (
    augment_gir_analysis_variables,
    compute_analysis_variables,
    extend_gir_var_labels,
    get_shock_dimension,
    get_states_to_shock,
    prepare_analysis_context,
)


def create_GIR_fn(econ_model, config, simul_policies=None, analysis_hooks=None):
    """
    Create a function to compute Generalized Impulse Responses (GIRs) for analysis variables.

    GIRs are computed by shocking the state indices configured for the selected model.

    Args:
        econ_model: Economic model instance
        config: Dictionary with parameters:
            - gir_n_draws: Number of points to sample from the active simulation window
            - gir_trajectory_length: Length of trajectories to simulate
            - ir_shock_sizes: List of shock sizes as percentages (e.g., [5, 10, 20])
            - states_to_shock: Optional list of state indices to shock
            - gir_seed: Random seed
            - use_gir: Boolean selecting GIR (`True`) vs stochastic-SS IR (`False`)
            - gir_symmetric_shocks: If True, positive shock is symmetric with negative
                                    (A_pos = 1/A_neg), not (1 + shock_size) (default: True)
        simul_policies: Simulation policies for extracting model-specific analysis context
                       (optional, can be passed in GIR_fn)

    Shock magnitudes:
        With gir_symmetric_shocks=True (default):
            - 20% negative: A drops to 0.80, shock = log(0.80) = -0.223
            - 20% positive: A rises to 1.25 (= 1/0.80), shock = log(1.25) = +0.223 (symmetric!)

        With gir_symmetric_shocks=False (legacy):
            - 20% negative: A drops to 0.80, shock = log(0.80) = -0.223
            - 20% positive: A rises to 1.20, shock = log(1.20) = +0.182 (NOT symmetric)

    Note: This function expects inputs from simulation_verbose_fn which returns states and policies
          in log deviation form (not normalized by standard deviations).
    """

    shock_dimension = get_shock_dimension(econ_model, analysis_hooks)

    def random_draws(simul_obs, n_draws, seed=0):
        """Sample random points from the active simulation sample."""
        n_simul = simul_obs.shape[0]
        key = random.PRNGKey(seed)
        replace = bool(n_draws > n_simul)
        draw_count = n_draws if replace else min(n_draws, n_simul)
        indices = random.choice(key, n_simul, shape=(draw_count,), replace=replace)
        obs_draws = simul_obs[indices, :]
        return obs_draws

    def create_counterfactual_state(state_logdev, state_idx, shock_size, shock_sign="neg"):
        """
        Create counterfactual initial state with one state component shocked.

        Args:
            state_logdev: State in log deviation form (not normalized)
            state_idx: Index of the state variable to shock
            shock_size: Size of shock to apply (as fraction, e.g., 0.2 for 20%)
            shock_sign: "neg" for negative shock (decreases TFP), "pos" for positive shock

        Returns:
            counterfactual_state_logdev: Modified state in log deviation form

        Shock implementation (with gir_symmetric_shocks=True, default):
            For a -20% shock (TFP drops to 80% of current level):
                A_neg = 1 - shock_size = 0.8
                In logs: adds log(0.8) ≈ -0.223 to the log level

            For a +20% SYMMETRIC shock (same magnitude in log space):
                A_pos = 1 / A_neg = 1 / 0.8 = 1.25
                In logs: adds -log(0.8) = log(1.25) ≈ +0.223 to the log level

        Shock implementation (with gir_symmetric_shocks=False, legacy):
            For a +20% shock (TFP rises to 120% of current level):
                A_pos = 1 + shock_size = 1.2
                In logs: adds log(1.2) ≈ +0.182 to the log level (NOT symmetric!)
        """
        # Convert logdevs to actual log levels
        state_notnorm = state_logdev + econ_model.state_ss

        # Check if symmetric shocks are enabled (default: True)
        symmetric_shocks = config.get("gir_symmetric_shocks", True)

        # Apply shock to specified state variable in log terms
        if shock_sign == "neg":
            # Negative shock: A_neg = 1 - shock_size (e.g., 0.8 for 20% shock)
            # Add log(A_neg) = log(1 - shock_size) to log level
            log_shock = jnp.log(1 - shock_size)
            state_counterfactual_notnorm = state_notnorm.at[state_idx].add(log_shock)
        else:  # positive shock
            if symmetric_shocks:
                # Symmetric positive shock: A_pos = 1 / A_neg = 1 / (1 - shock_size)
                # Add log(A_pos) = -log(1 - shock_size) to log level (same magnitude, opposite sign)
                log_shock = -jnp.log(1 - shock_size)
            else:
                # Legacy positive shock: A_pos = 1 + shock_size (NOT symmetric)
                log_shock = jnp.log(1 + shock_size)
            state_counterfactual_notnorm = state_notnorm.at[state_idx].add(log_shock)

        # Convert back to logdevs
        state_counterfactual_logdev = state_counterfactual_notnorm - econ_model.state_ss

        return state_counterfactual_logdev

    def simul_trajectory_analysis_variables(
        econ_model,
        train_state,
        shocks,
        obs_init_logdev,
        analysis_context,
        var_labels,
        state_idx,
    ):
        """
        Simulate full trajectory and return analysis variables at each step.

        Args:
            econ_model: Economic model
            train_state: Trained neural network state
            shocks: Sequence of shocks
            obs_init_logdev: Initial observation in log deviation form
            analysis_context: Model-specific context for computing analysis variables
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
            analysis_vars_dict = compute_analysis_variables(
                econ_model=econ_model,
                state_logdev=obs_logdev,
                policy_logdev=policy_logdev,
                analysis_context=analysis_context,
                analysis_hooks=analysis_hooks,
            )
            analysis_vars_dict = augment_gir_analysis_variables(
                analysis_vars_dict=analysis_vars_dict,
                obs_logdev=obs_logdev,
                policy_logdev=policy_logdev,
                state_idx=state_idx,
                econ_model=econ_model,
                config=config,
                analysis_hooks=analysis_hooks,
            )
            analysis_vars_array = jnp.array([analysis_vars_dict[label] for label in var_labels])
            return next_obs_logdev, analysis_vars_array

        final_obs_logdev, trajectory_analysis_variables = lax.scan(step, obs_init_logdev, shocks)

        # Compute analysis variables for the final observation (using last policy)
        final_obs_normalized = final_obs_logdev / econ_model.state_sd
        final_policy_normalized = train_state.apply_fn(train_state.params, final_obs_normalized)
        final_policy_logdev = final_policy_normalized * econ_model.policies_sd
        final_analysis_vars_dict = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=final_obs_logdev,
            policy_logdev=final_policy_logdev,
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )
        final_analysis_vars_dict = augment_gir_analysis_variables(
            analysis_vars_dict=final_analysis_vars_dict,
            obs_logdev=final_obs_logdev,
            policy_logdev=final_policy_logdev,
            state_idx=state_idx,
            econ_model=econ_model,
            config=config,
            analysis_hooks=analysis_hooks,
        )
        final_analysis_vars_array = jnp.array([final_analysis_vars_dict[label] for label in var_labels])

        # Add final analysis variables to trajectory
        trajectory_analysis_variables_full = jnp.concatenate(
            [trajectory_analysis_variables, final_analysis_vars_array[None, :]], axis=0
        )

        return trajectory_analysis_variables_full

    def compute_state_GIR(simul_obs, train_state, state_idx, analysis_context, var_labels, shock_size, shock_sign):
        """
        Compute GIR for a specific TFP state using analysis variables.

        Args:
            simul_obs: Active simulation observations (in logdev form)
            train_state: Trained neural network state
            state_idx: Index of the state variable to shock
            analysis_context: Model-specific context for computing analysis variables
            var_labels: List of variable labels in consistent order
            shock_size: Size of shock (fraction, e.g., 0.2 for 20%)
            shock_sign: "neg" or "pos"

        Returns:
            gir_analysis_variables: Averaged impulse response for analysis variables [T+1, n_analysis_vars]
        """
        # Sample points from the active common-shock simulation window.
        sample_points = random_draws(simul_obs, config["gir_n_draws"], config["gir_seed"])

        # Function to compute impulse response for a single initial point
        def single_point_IR(_i, obs_init_logdev):
            # Keep paths deterministic after the initial perturbation.
            zero_shocks = jnp.zeros((config["gir_trajectory_length"], shock_dimension))

            # Original trajectory analysis variables
            traj_analysis_vars_orig = simul_trajectory_analysis_variables(
                econ_model,
                train_state,
                zero_shocks,
                obs_init_logdev,
                analysis_context,
                var_labels,
                state_idx=state_idx,
            )

            # Create counterfactual initial state with specified shock sign
            obs_counterfactual_logdev = create_counterfactual_state(obs_init_logdev, state_idx, shock_size, shock_sign)

            # Counterfactual trajectory analysis variables (using same shocks)
            traj_analysis_vars_counter = simul_trajectory_analysis_variables(
                econ_model,
                train_state,
                zero_shocks,
                obs_counterfactual_logdev,
                analysis_context,
                var_labels,
                state_idx=state_idx,
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

    def compute_ir_from_state(
        initial_state_logdev,
        train_state,
        state_idx,
        analysis_context,
        var_labels,
        shock_size,
        shock_sign,
    ):
        """
        Compute IR from a specific initial state (e.g., stochastic steady state).

        Unlike GIR which averages over many draws from the active simulation sample,
        this computes the IR from a single deterministic starting point.

        Args:
            initial_state_logdev: Initial state in log deviation form
            train_state: Trained neural network state
            state_idx: Index of the state variable to shock
            analysis_context: Model-specific context for computing analysis variables
            var_labels: List of variable labels in consistent order
            shock_size: Size of shock (fraction, e.g., 0.2 for 20%)
            shock_sign: "neg" or "pos"

        Returns:
            ir_analysis_variables: Impulse response for analysis variables [T+1, n_analysis_vars]
        """
        # Generate zero shocks for the trajectory (no uncertainty after initial shock)
        zero_shocks = jnp.zeros((config["gir_trajectory_length"], shock_dimension))

        # Original trajectory (no shock)
        traj_analysis_vars_orig = simul_trajectory_analysis_variables(
            econ_model,
            train_state,
            zero_shocks,
            initial_state_logdev,
            analysis_context,
            var_labels,
            state_idx=state_idx,
        )

        # Create counterfactual initial state with specified shock
        obs_counterfactual_logdev = create_counterfactual_state(initial_state_logdev, state_idx, shock_size, shock_sign)

        # Counterfactual trajectory (with initial shock, then no shocks)
        traj_analysis_vars_counter = simul_trajectory_analysis_variables(
            econ_model,
            train_state,
            zero_shocks,
            obs_counterfactual_logdev,
            analysis_context,
            var_labels,
            state_idx=state_idx,
        )

        # Impulse response is difference
        ir_analysis_variables = traj_analysis_vars_counter - traj_analysis_vars_orig

        return ir_analysis_variables

    def _resolve_requested_ir_methods(stoch_ss_state_logdev):
        use_gir = config.get("use_gir")
        if use_gir is not None:
            if bool(use_gir):
                return ["GIR"]
            if stoch_ss_state_logdev is None:
                raise ValueError(
                    "config['use_gir']=False requires a stochastic steady state so the "
                    "stochastic-steady-state IR can be computed."
                )
            return ["IR_stoch_ss"]

        configured_methods = config.get("ir_methods")
        if configured_methods is None:
            configured_methods = [config.get("ir_method", "GIR")]
        elif isinstance(configured_methods, str):
            configured_methods = [configured_methods]

        resolved_methods = []
        for method in configured_methods:
            if method and method not in resolved_methods:
                resolved_methods.append(method)
        return resolved_methods or ["GIR"]

    def GIR_fn(simul_obs, train_state, simul_policies_data=None, stoch_ss_state_logdev=None):
        """
        Main GIR function that computes TFP impulse responses for specified sectors.
        Computes GIRs for both positive and negative TFP shocks, and for multiple shock sizes.

        Args:
            simul_obs: Active simulation observations (in logdev form)
            train_state: Trained neural network state
            simul_policies_data: Simulation policies for extracting price weights (in logdev form)
            stoch_ss_state_logdev: Optional stochastic steady state (in logdev form).
                                   If provided, also computes IRs from this state.

        Returns:
            gir_results: Dictionary with structure:
                {state_name: {
                    "state_idx": int,  # TFP state index (n_sectors + sector_idx)
                    "pos_5": {"gir_analysis_variables": {...}},   # +5% TFP shock (GIR)
                    "neg_5": {"gir_analysis_variables": {...}},   # -5% TFP shock (GIR)
                    "pos_5_stochss": {"gir_analysis_variables": {...}},   # +5% from stoch SS
                    "neg_5_stochss": {"gir_analysis_variables": {...}},   # -5% from stoch SS
                    ...
                }}
        """
        # Use provided simul_policies or the one passed during creation
        if simul_policies_data is None:
            simul_policies_data = simul_policies

        if simul_policies_data is None:
            raise ValueError("simul_policies must be provided either during create_GIR_fn or GIR_fn call")

        analysis_context = prepare_analysis_context(
            econ_model=econ_model,
            simul_obs=simul_obs,
            simul_policies=simul_policies_data,
            config=config,
            analysis_hooks=analysis_hooks,
        )

        # Get variable labels from a single analysis call
        first_analysis_vars = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=simul_obs[0],
            policy_logdev=simul_policies_data[0],
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )
        var_labels = extend_gir_var_labels(
            var_labels=list(first_analysis_vars.keys()),
            econ_model=econ_model,
            config=config,
            analysis_hooks=analysis_hooks,
        )

        # Determine which states to shock
        states_to_shock = get_states_to_shock(config=config, econ_model=econ_model, analysis_hooks=analysis_hooks)

        # Get shock sizes from config (as percentages, e.g., [5, 10, 20])
        ir_shock_sizes = config.get("ir_shock_sizes", [5, 10, 20])

        # `use_gir` is the current selector between GIR and stochastic-SS IRs.
        # Fall back to the legacy `ir_methods`/`ir_method` config only when needed.
        configured_methods = _resolve_requested_ir_methods(stoch_ss_state_logdev)

        compute_gir = "GIR" in configured_methods
        compute_stochss_irs = "IR_stoch_ss" in configured_methods

        gir_results = {}

        # Calculate total computations (2 for pos/neg per method).
        n_ir_types = (2 if compute_gir else 0) + (2 if compute_stochss_irs else 0)
        total_computations = len(states_to_shock) * len(ir_shock_sizes) * n_ir_types
        current_computation = 0

        # Compute GIR for each specified state
        for state_idx in states_to_shock:
            # Create state label
            state_name = f"state_{state_idx}"
            if hasattr(econ_model, "state_labels") and state_idx < len(econ_model.state_labels):
                state_name = econ_model.state_labels[state_idx]

            gir_results[state_name] = {"state_idx": state_idx}

            # Compute IR for each shock size and sign
            for shock_size_pct in ir_shock_sizes:
                shock_size = shock_size_pct / 100.0  # Convert percentage to fraction

                for shock_sign in ["pos", "neg"]:
                    # Compute GIR averaged over whichever nonlinear reference sample was supplied.
                    if compute_gir:
                        current_computation += 1
                        print(
                            f"      GIR [{current_computation}/{total_computations}]: "
                            f"{state_name} (state {state_idx}), {shock_sign}_{shock_size_pct}%",
                            flush=True,
                        )

                        gir_analysis_vars_array = compute_state_GIR(
                            simul_obs,
                            train_state,
                            state_idx,
                            analysis_context,
                            var_labels,
                            shock_size,
                            shock_sign,
                        )

                        gir_analysis_vars_dict = {
                            label: gir_analysis_vars_array[:, i] for i, label in enumerate(var_labels)
                        }

                        key = f"{shock_sign}_{shock_size_pct}"
                        gir_results[state_name][key] = {"gir_analysis_variables": gir_analysis_vars_dict}

                    # Compute IR from stochastic steady state if requested.
                    if compute_stochss_irs:
                        current_computation += 1
                        print(
                            f"      IR_stoch_ss [{current_computation}/{total_computations}]: "
                            f"{state_name} (state {state_idx}), {shock_sign}_{shock_size_pct}%",
                            flush=True,
                        )

                        stochss_ir_array = compute_ir_from_state(
                            stoch_ss_state_logdev,
                            train_state,
                            state_idx,
                            analysis_context,
                            var_labels,
                            shock_size,
                            shock_sign,
                        )

                        stochss_ir_dict = {label: stochss_ir_array[:, i] for i, label in enumerate(var_labels)}

                        # Store with key like "pos_5_stochss", "neg_10_stochss", etc.
                        stochss_key = f"{shock_sign}_{shock_size_pct}_stochss"
                        gir_results[state_name][stochss_key] = {"gir_analysis_variables": stochss_ir_dict}

        return gir_results

    return GIR_fn
