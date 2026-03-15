import importlib
from types import ModuleType
from typing import Any, Dict, Optional

from jax import random
from jax import numpy as jnp


def load_model_analysis_hooks(model_dir: str) -> Optional[ModuleType]:
    """Load optional model-specific analysis hooks."""
    module_name = f"DEQN.econ_models.{model_dir}.analysis_hooks"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def apply_model_config_defaults(config: Dict[str, Any], analysis_hooks: Optional[ModuleType]) -> Dict[str, Any]:
    """Apply model-local defaults without overriding explicit user config."""
    if analysis_hooks is None:
        return config

    defaults = getattr(analysis_hooks, "DEFAULT_ANALYSIS_CONFIG", None)
    if not defaults:
        return config

    merged = dict(config)
    for key, value in defaults.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    return merged


def prepare_analysis_context(
    econ_model: Any,
    simul_obs: jnp.ndarray,
    simul_policies: jnp.ndarray,
    config: Dict[str, Any],
    analysis_hooks: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    """Prepare optional model-specific context for analysis variables."""
    if analysis_hooks is not None and hasattr(analysis_hooks, "prepare_analysis_context"):
        return analysis_hooks.prepare_analysis_context(
            econ_model=econ_model,
            simul_obs=simul_obs,
            simul_policies=simul_policies,
            config=config,
        )
    return {}


def compute_analysis_variables(
    econ_model: Any,
    state_logdev: jnp.ndarray,
    policy_logdev: jnp.ndarray,
    analysis_context: Optional[Dict[str, Any]] = None,
    analysis_hooks: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    """Compute analysis variables using either model hooks or the model directly."""
    context = analysis_context or {}

    if analysis_hooks is not None and hasattr(analysis_hooks, "compute_analysis_variables"):
        return analysis_hooks.compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=state_logdev,
            policy_logdev=policy_logdev,
            analysis_context=context,
        )

    if not hasattr(econ_model, "get_analysis_variables"):
        raise AttributeError(
            f"Model '{type(econ_model).__name__}' does not implement get_analysis_variables() "
            "and no analysis_hooks.compute_analysis_variables() was provided."
        )

    if context:
        return econ_model.get_analysis_variables(state_logdev, policy_logdev, **context)
    return econ_model.get_analysis_variables(state_logdev, policy_logdev)


def extend_gir_var_labels(
    var_labels: list[str],
    econ_model: Any,
    config: Dict[str, Any],
    analysis_hooks: Optional[ModuleType] = None,
) -> list[str]:
    """Allow model hooks to append extra GIR-only labels."""
    extended = list(var_labels)
    if analysis_hooks is not None and hasattr(analysis_hooks, "extend_gir_var_labels"):
        return analysis_hooks.extend_gir_var_labels(
            var_labels=extended,
            econ_model=econ_model,
            config=config,
        )
    return extended


def augment_gir_analysis_variables(
    analysis_vars_dict: Dict[str, Any],
    obs_logdev: jnp.ndarray,
    policy_logdev: jnp.ndarray,
    state_idx: int,
    econ_model: Any,
    config: Dict[str, Any],
    analysis_hooks: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    """Allow model hooks to inject extra IR variables derived from raw state/policy blocks."""
    if analysis_hooks is not None and hasattr(analysis_hooks, "augment_gir_analysis_variables"):
        return analysis_hooks.augment_gir_analysis_variables(
            analysis_vars_dict=dict(analysis_vars_dict),
            obs_logdev=obs_logdev,
            policy_logdev=policy_logdev,
            state_idx=state_idx,
            econ_model=econ_model,
            config=config,
        )
    return analysis_vars_dict


def get_states_to_shock(
    config: Dict[str, Any],
    econ_model: Any,
    analysis_hooks: Optional[ModuleType] = None,
) -> list[int]:
    """Resolve which states should be shocked for GIR analysis."""
    configured_states = config.get("states_to_shock")
    if configured_states is not None:
        return list(configured_states)

    if analysis_hooks is not None and hasattr(analysis_hooks, "get_states_to_shock"):
        return list(analysis_hooks.get_states_to_shock(config=config, econ_model=econ_model))

    return list(range(econ_model.dim_states))


def get_shock_dimension(econ_model: Any, analysis_hooks: Optional[ModuleType] = None) -> int:
    """Infer the dimension of the model shock vector."""
    if analysis_hooks is not None and hasattr(analysis_hooks, "get_shock_dimension"):
        return int(analysis_hooks.get_shock_dimension(econ_model=econ_model))

    sample_shock = econ_model.sample_shock(random.PRNGKey(0))
    shock_array = jnp.atleast_1d(sample_shock)
    return int(shock_array.shape[-1])


def run_model_postprocess(
    analysis_hooks: Optional[ModuleType],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run optional model-specific post-processing after generic simulations."""
    if analysis_hooks is None or not hasattr(analysis_hooks, "postprocess_analysis"):
        return {}

    results = analysis_hooks.postprocess_analysis(**kwargs)
    return results or {}
