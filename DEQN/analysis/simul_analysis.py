import jax
from jax import lax, random
from jax import numpy as jnp


def create_episode_simulation_fn_verbose(econ_model, config):
    """Create simulation function that returns both observations and policies in logdevs."""

    def sample_epis_obs_and_policies(train_state, epis_rng):
        """Sample observations and policies for an episode."""
        init_obs = econ_model.initial_state(epis_rng, config["init_range"])
        period_rngs = random.split(epis_rng, config["periods_per_epis"])

        def period_step(env_obs, period_rng):
            policy = train_state.apply_fn(train_state.params, env_obs)
            period_shock = config["simul_vol_scale"] * econ_model.sample_shock(period_rng)
            obs_next = econ_model.step(env_obs, policy, period_shock)
            obs_next_logdev = obs_next * econ_model.state_sd
            policy_logdev = policy * econ_model.policies_sd
            return obs_next, (obs_next_logdev, policy_logdev)

        _, (epis_obs, epis_policies) = lax.scan(period_step, init_obs, jnp.stack(period_rngs))
        return epis_obs, epis_policies

    return sample_epis_obs_and_policies


def simulation_analysis(train_state, econ_model, analysis_config, simulation_fn):
    """
    Run parallel simulations and compute analysis variables.

    Parameters:
    -----------
    train_state : Training state with trained neural network parameters
    econ_model : Economic model instance
    analysis_config : Dictionary containing simulation configuration
    simulation_fn : JIT-compiled simulation function (typically create_episode_simulation_fn_verbose)

    Returns:
    --------
    simul_obs : JAX array of shape (n_seeds * n_periods, n_state_vars)
        Combined simulation observations after burn-in (in log deviation form)
    simul_policies : JAX array of shape (n_seeds * n_periods, n_policy_vars)
        Combined simulation policies after burn-in (in log deviation form)
    simul_analysis_variables : JAX array of shape (n_seeds * n_periods, n_analysis_vars)
        Computed analysis variables for each observation

    Note: This function expects simulation_fn to return states and policies in log deviation form
          (not normalized by standard deviations), which is the case for create_episode_simulation_fn_verbose.
    """
    print(f"  Generating simulation data with {analysis_config['n_simul_seeds']} seeds...")

    # Generate random seeds for parallel simulations
    base_rng = random.PRNGKey(analysis_config["simul_seed"])
    episode_rngs = random.split(base_rng, analysis_config["n_simul_seeds"])

    # Run simulations in parallel using vmap
    multi_simulation_fn = jax.vmap(simulation_fn, in_axes=(None, 0))
    simul_state_multi, simul_policies_multi = multi_simulation_fn(train_state, episode_rngs)

    # Extract burn-in period: shape (n_seeds, T, n_obs) -> (n_seeds, T-burn_in, n_obs)
    simul_state_multi = simul_state_multi[:, analysis_config["burn_in_periods"] :, :]
    simul_policies_multi = simul_policies_multi[:, analysis_config["burn_in_periods"] :, :]

    # Reshape to combine seed and time dimensions: (n_seeds, T-burn_in, n_obs) -> (n_seeds*(T-burn_in), n_obs)
    n_seeds, n_periods, n_state_vars = simul_state_multi.shape
    _, _, n_policy_vars = simul_policies_multi.shape
    simul_obs = simul_state_multi.reshape(n_seeds * n_periods, n_state_vars)
    simul_policies = simul_policies_multi.reshape(n_seeds * n_periods, n_policy_vars)

    # Validation test: the last observation should be reasonable (logdev should be within reasonable bounds)
    max_dev = jnp.max(jnp.abs(simul_obs[-1, :]))
    print(f"  Max log deviation in last obs of simulation: {max_dev:.6f}")
    assert max_dev < 10, f"Last observation too large: {max_dev:.6f}"

    print(f"  Using {simul_obs.shape[0]} total observations ({n_seeds} seeds × {n_periods} periods) for statistics")

    # Get mean states and policies from ergodic distribution to construct analysis variables
    simul_policies_mean = jnp.mean(simul_policies, axis=0)
    P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
    Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
    Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

    # Get analysis variables for first observation to extract labels
    first_analysis_vars = econ_model.get_analysis_variables(simul_obs[0], simul_policies[0], P_mean, Pk_mean, Pm_mean)
    var_labels = list(first_analysis_vars.keys())

    # Vectorized computation returns dictionary per observation
    # We need to convert to array format for vmap, then reconstruct dictionary
    def get_vars_as_array(obs, pol, P, Pk, Pm):
        var_dict = econ_model.get_analysis_variables(obs, pol, P, Pk, Pm)
        return jnp.array([var_dict[label] for label in var_labels])

    # Get analysis variables as array (n_obs, n_vars)
    simul_analysis_vars_array = jax.vmap(get_vars_as_array, in_axes=(0, 0, None, None, None))(
        simul_obs, simul_policies, P_mean, Pk_mean, Pm_mean
    )

    # Convert to dictionary format: {label: array of values across time}
    simul_analysis_variables = {label: simul_analysis_vars_array[:, i] for i, label in enumerate(var_labels)}

    return simul_obs, simul_policies, simul_analysis_variables
