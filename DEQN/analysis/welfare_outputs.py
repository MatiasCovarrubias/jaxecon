import jax
from jax import random

import jax.numpy as jnp

from DEQN.analysis.io import _normalize_dynare_full_simul
from DEQN.analysis.simul_analysis import create_loglinear_episode_utility_fn, simulate_ergodic_utilities

WELFARE_BOTH_RECENTERED_LABEL = "C and L recentered at determ SS"
WELFARE_L_FIXED_AT_DSS_LABEL = "L fixed at determ SS"


def _compute_welfare_cost_from_utilities(*, econ_model, welfare_fn, welfare_ss, utilities, rng_key):
    welfare = welfare_fn(utilities, welfare_ss, rng_key)
    return -econ_model.consumption_equivalent(welfare) * 100


def _compute_welfare_cost_from_sample(*, econ_model, welfare_fn, welfare_ss, policies_logdev, config_dict):
    simul_utilities = jax.vmap(econ_model.utility_from_policies)(policies_logdev)
    return _compute_welfare_cost_from_utilities(
        econ_model=econ_model,
        welfare_fn=welfare_fn,
        welfare_ss=welfare_ss,
        utilities=simul_utilities,
        rng_key=random.PRNGKey(config_dict["welfare_seed"]),
    )


def _recenter_logdev_path_to_dss(path_logdev):
    return path_logdev - jnp.mean(path_logdev)


def _compute_counterfactual_utilities_from_sample(
    *,
    econ_model,
    policies_logdev,
    recenter_consumption=False,
    recenter_labor=False,
    fix_labor_at_dss=False,
):
    if not recenter_consumption and not recenter_labor and not fix_labor_at_dss:
        raise ValueError("At least one welfare counterfactual adjustment must be requested.")
    if recenter_labor and fix_labor_at_dss:
        raise ValueError("Labor cannot be both recentered and fixed at the deterministic steady state.")

    consumption_logdev = policies_logdev[:, econ_model.c_util_idx]
    labor_logdev = policies_logdev[:, econ_model.l_util_idx]

    if recenter_consumption:
        consumption_logdev = _recenter_logdev_path_to_dss(consumption_logdev)
    if recenter_labor:
        labor_logdev = _recenter_logdev_path_to_dss(labor_logdev)
    if fix_labor_at_dss:
        labor_logdev = jnp.zeros_like(labor_logdev)

    consumption_level = jnp.exp(consumption_logdev + econ_model.policies_ss[econ_model.c_util_idx])
    labor_level = jnp.exp(labor_logdev + econ_model.policies_ss[econ_model.l_util_idx])
    labor_exponent = 1 + econ_model.eps_l ** (-1)
    labor_disutility = econ_model.theta * (1 / (1 + econ_model.eps_l ** (-1))) * labor_level**labor_exponent
    utility_intratemp = consumption_level - labor_disutility

    if float(jnp.min(utility_intratemp)) <= 0:
        raise ValueError("Counterfactual intratemporal utility became non-positive.")

    return econ_model._utility_from_intratemp_level(utility_intratemp)


def _compute_counterfactual_welfare_cost_from_sample(
    *,
    econ_model,
    welfare_fn,
    welfare_ss,
    policies_logdev,
    config_dict,
    recenter_consumption=False,
    recenter_labor=False,
    fix_labor_at_dss=False,
):
    simul_utilities = _compute_counterfactual_utilities_from_sample(
        econ_model=econ_model,
        policies_logdev=policies_logdev,
        recenter_consumption=recenter_consumption,
        recenter_labor=recenter_labor,
        fix_labor_at_dss=fix_labor_at_dss,
    )
    return _compute_welfare_cost_from_utilities(
        econ_model=econ_model,
        welfare_fn=welfare_fn,
        welfare_ss=welfare_ss,
        utilities=simul_utilities,
        rng_key=random.PRNGKey(config_dict["welfare_seed"]),
    )


def _welfare_cost_from_dynare_simul(
    simul_data,
    method_name,
    state_ss,
    policies_ss,
    econ_model,
    welfare_fn,
    welfare_ss,
    config_dict,
):
    """Compute consumption-equivalent welfare cost from the canonical active Dynare sample."""
    if simul_data is None:
        return None
    simul_matrix = _normalize_dynare_full_simul(simul_data, state_ss, policies_ss)
    n_state_vars = state_ss.shape[0]
    policies_logdev = simul_matrix[n_state_vars:, :].T

    if policies_logdev.shape[0] == 0:
        print(f"  ⚠ Skipping welfare for {method_name}: active sample is empty.")
        return None

    method_seed = sum(ord(c) for c in method_name)
    return _compute_welfare_cost_from_utilities(
        econ_model=econ_model,
        welfare_fn=welfare_fn,
        welfare_ss=welfare_ss,
        utilities=jax.vmap(econ_model.utility_from_policies)(policies_logdev),
        rng_key=random.PRNGKey(config_dict["welfare_seed"] + method_seed),
    )


def _welfare_cost_from_loglinear_long_simulation(
    *,
    method_name,
    state_transition_matrix,
    state_shock_matrix,
    policy_state_matrix,
    policy_shock_matrix,
    econ_model,
    welfare_fn,
    welfare_ss,
    config_dict,
):
    episode_utility_fn = jax.jit(
        create_loglinear_episode_utility_fn(
            econ_model=econ_model,
            config=config_dict,
            state_transition_matrix=state_transition_matrix,
            state_shock_matrix=state_shock_matrix,
            policy_state_matrix=policy_state_matrix,
            policy_shock_matrix=policy_shock_matrix,
        )
    )
    simul_utilities = simulate_ergodic_utilities(
        analysis_config=config_dict,
        episode_utility_fn=episode_utility_fn,
        label=f"Long ergodic {method_name} simulation",
    )
    if simul_utilities.shape[0] == 0:
        print(f"  ⚠ Skipping welfare for {method_name}: retained long simulation sample is empty.")
        return None

    method_seed = sum(ord(c) for c in method_name)
    return _compute_welfare_cost_from_utilities(
        econ_model=econ_model,
        welfare_fn=welfare_fn,
        welfare_ss=welfare_ss,
        utilities=simul_utilities,
        rng_key=random.PRNGKey(config_dict["welfare_seed"] + method_seed),
    )
