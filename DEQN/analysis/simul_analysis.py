import jax
from jax import numpy as jnp
from jax import random


def simulation_analysis(train_state, econ_model, analysis_config, simulation_fn):
    """
    Run parallel simulations and compute aggregates.

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
    simul_aggregates : JAX array of shape (n_seeds * n_periods, n_aggregates)
        Computed aggregate variables for each observation

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

    # Get mean states and policies from ergodic distribution to construct aggregates
    simul_policies_mean = jnp.mean(simul_policies, axis=0)
    P_mean = simul_policies_mean[8 * econ_model.n_sectors : 9 * econ_model.n_sectors]
    Pk_mean = simul_policies_mean[2 * econ_model.n_sectors : 3 * econ_model.n_sectors]
    Pm_mean = simul_policies_mean[3 * econ_model.n_sectors : 4 * econ_model.n_sectors]

    # Get aggregates for simulation data
    simul_aggregates = jax.vmap(econ_model.get_aggregates, in_axes=(0, 0, None, None, None))(
        simul_obs, simul_policies, P_mean, Pk_mean, Pm_mean
    )

    return simul_obs, simul_policies, simul_aggregates
