import jax
from jax import lax, random
from jax import numpy as jnp

from DEQN.analysis.model_hooks import compute_analysis_variables, prepare_analysis_context


def _compute_analysis_dataset(
    *,
    econ_model,
    simul_obs,
    simul_policies,
    analysis_config,
    analysis_hooks=None,
    analysis_context_override=None,
):
    """Build model-specific analysis variables for a simulation sample."""
    if simul_obs.shape[0] == 0:
        raise ValueError("Cannot compute analysis variables for an empty simulation sample.")

    if analysis_context_override is None:
        analysis_context = prepare_analysis_context(
            econ_model=econ_model,
            simul_obs=simul_obs,
            simul_policies=simul_policies,
            config=analysis_config,
            analysis_hooks=analysis_hooks,
        )
    else:
        analysis_context = analysis_context_override

    first_analysis_vars = compute_analysis_variables(
        econ_model=econ_model,
        state_logdev=simul_obs[0],
        policy_logdev=simul_policies[0],
        analysis_context=analysis_context,
        analysis_hooks=analysis_hooks,
    )
    var_labels = list(first_analysis_vars.keys())

    def get_vars_as_array(obs, pol):
        var_dict = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=obs,
            policy_logdev=pol,
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )
        return jnp.array([var_dict[label] for label in var_labels])

    simul_analysis_vars_array = jax.vmap(get_vars_as_array)(simul_obs, simul_policies)
    simul_analysis_variables = {label: simul_analysis_vars_array[:, i] for i, label in enumerate(var_labels)}
    return simul_analysis_variables, analysis_context


def compute_analysis_dataset_with_context(
    *,
    econ_model,
    simul_obs,
    simul_policies,
    analysis_config,
    analysis_context,
    analysis_hooks=None,
):
    """Build analysis variables using a caller-supplied aggregation context."""
    return _compute_analysis_dataset(
        econ_model=econ_model,
        simul_obs=simul_obs,
        simul_policies=simul_policies,
        analysis_config=analysis_config,
        analysis_hooks=analysis_hooks,
        analysis_context_override=analysis_context,
    )


def create_episode_simulation_fn_verbose(econ_model, config):
    """
    Create simulation function that returns observations and policies in log-deviation form.

    The model uses standardized state and policy internally (logdev / state_sd, logdev / policies_sd).
    Here we denormalize before returning so that simul_obs and simul_policies are in raw log
    deviations, as expected by get_analysis_variables, utility_from_policies, and ergodic price
    computation (log level = logdev + ss, so logdev = standardized * sd).
    """

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


def create_shock_path_simulation_fn(econ_model):
    """
    Create simulation function that follows a provided shock path.

    The returned states and policies are in raw log-deviation form, matching
    the convention used by the analysis pipeline.
    """

    def sample_obs_and_policies_given_shocks(train_state, initial_obs_logdev, shock_path):
        """Simulate one path from an explicit sequence of shocks."""

        def period_step(obs_logdev, period_shock):
            obs_normalized = obs_logdev / econ_model.state_sd
            policy_normalized = train_state.apply_fn(train_state.params, obs_normalized)
            next_obs_normalized = econ_model.step(obs_normalized, policy_normalized, period_shock)
            next_obs_logdev = next_obs_normalized * econ_model.state_sd
            policy_logdev = policy_normalized * econ_model.policies_sd
            return next_obs_logdev, (next_obs_logdev, policy_logdev)

        _, (epis_obs, epis_policies) = lax.scan(period_step, initial_obs_logdev, shock_path)
        return epis_obs, epis_policies

    return sample_obs_and_policies_given_shocks


def simulation_analysis(train_state, econ_model, analysis_config, simulation_fn, analysis_hooks=None):
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
    simul_analysis_variables : dict
        Computed analysis variables for each observation, keyed by label
    analysis_context : dict
        Optional model-specific context used to build analysis variables

    Note: This function expects simulation_fn to return states and policies in log deviation form
          (not normalized by standard deviations), which is the case for create_episode_simulation_fn_verbose.
    """
    base_rng = random.PRNGKey(analysis_config["simul_seed"])
    episode_rngs = random.split(base_rng, analysis_config["n_simul_seeds"])

    multi_simulation_fn = jax.vmap(simulation_fn, in_axes=(None, 0))
    simul_state_multi, simul_policies_multi = multi_simulation_fn(train_state, episode_rngs)

    simul_state_multi = simul_state_multi[:, analysis_config["burn_in_periods"] :, :]
    simul_policies_multi = simul_policies_multi[:, analysis_config["burn_in_periods"] :, :]

    n_seeds, n_periods, n_state_vars = simul_state_multi.shape
    _, _, n_policy_vars = simul_policies_multi.shape
    simul_obs = simul_state_multi.reshape(n_seeds * n_periods, n_state_vars)
    simul_policies = simul_policies_multi.reshape(n_seeds * n_periods, n_policy_vars)

    max_dev = jnp.max(jnp.abs(simul_obs[-1, :]))
    assert max_dev < 10, f"Last observation too large: {max_dev:.6f}"

    n_obs = simul_obs.shape[0]
    print(f"    Simulation: {n_obs:,} obs ({n_seeds} seeds × {n_periods} periods)", flush=True)

    simul_analysis_variables, analysis_context = _compute_analysis_dataset(
        econ_model=econ_model,
        simul_obs=simul_obs,
        simul_policies=simul_policies,
        analysis_config=analysis_config,
        analysis_hooks=analysis_hooks,
    )

    return simul_obs, simul_policies, simul_analysis_variables, analysis_context


def simulation_analysis_with_shocks(
    train_state,
    econ_model,
    shock_path,
    simulation_fn,
    active_start,
    active_end,
    analysis_config,
    analysis_hooks=None,
    initial_obs_logdev=None,
    label="Common-shock simulation",
):
    """
    Run one simulation using a provided shock path and compute analysis variables.

    The active window `[active_start:active_end)` is the canonical analysis sample.
    The full path is still returned so downstream code can keep the explicit
    burn-in / active / burn-out structure when needed.
    """
    if initial_obs_logdev is None:
        initial_obs_logdev = jnp.zeros_like(econ_model.state_ss)

    simul_obs_full, simul_policies_full = simulation_fn(train_state, initial_obs_logdev, shock_path)

    if active_start < 0 or active_end > simul_obs_full.shape[0] or active_start >= active_end:
        raise ValueError(
            f"Invalid active window [{active_start}, {active_end}) for simulation of length {simul_obs_full.shape[0]}."
        )

    simul_obs = simul_obs_full[active_start:active_end]
    simul_policies = simul_policies_full[active_start:active_end]

    print(
        f"    {label}: {simul_obs.shape[0]:,} active obs from {simul_obs_full.shape[0]:,} total periods",
        flush=True,
    )

    simul_analysis_variables, analysis_context = _compute_analysis_dataset(
        econ_model=econ_model,
        simul_obs=simul_obs,
        simul_policies=simul_policies,
        analysis_config=analysis_config,
        analysis_hooks=analysis_hooks,
    )

    return (
        simul_obs,
        simul_policies,
        simul_analysis_variables,
        analysis_context,
        simul_obs_full,
        simul_policies_full,
    )


# Legacy long-ergodic simulation path kept for later reuse:
# def legacy_long_ergodic_analysis(train_state, econ_model, analysis_config, simulation_fn, analysis_hooks=None):
#     return simulation_analysis(
#         train_state=train_state,
#         econ_model=econ_model,
#         analysis_config=analysis_config,
#         simulation_fn=simulation_fn,
#         analysis_hooks=analysis_hooks,
#     )
