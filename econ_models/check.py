"""Checks for the shared model contract.

Every model provides ``params``, ``discount_rate``, ``state_ss``, ``state_sd``,
``control_ss``, ``control_sd``, ``steady_state``, ``transition``,
``sample_shock``, and ``initial_state``. Scales are positive, so a caller can
normalize, and the identity scale is allowed. Methods receive the model's own
coordinates.

``utility(state, control)`` is optional. When it is present it is a scalar
period utility and welfare analysis can use it. APG requires it.

DEQN and time iteration require ``expectation(state, control)``, the realized
continuation at one state, and ``residuals(state, control, expectation)``, a
vector with one unsquared entry per equation. Time iteration also requires
``n_endogenous == 1``, with the remaining state entries matching the shock.
"""

import jax
from jax import numpy as jnp
from jax import random

_ALGORITHMS = ("apg", "deqn", "time_iteration")


def check_model(model):
    """Raise ``ValueError`` when ``model`` misses the shared contract."""
    state_ss, _, control_ss, _ = _scales(model)
    _require(0.0 < float(model.discount_rate) < 1.0, "discount_rate must lie in (0, 1)")
    _finite_tree(model.params, "params")

    steady = model.steady_state()
    state = jnp.asarray(steady.state)
    control = jnp.asarray(steady.control)
    _match(state, state_ss, "steady_state().state")
    _match(control, control_ss, "steady_state().control")

    shock_rng, init_rng = random.split(random.PRNGKey(0))
    shock = jnp.asarray(model.sample_shock(shock_rng))
    _require(shock.ndim == 1 and shock.shape[0] > 0, "sample_shock must return a non-empty vector")
    _finite(shock, "sample_shock")

    next_state = jnp.asarray(model.transition(state, control, jnp.zeros_like(shock)))
    _match(next_state, state, "transition at the steady state")
    if not jnp.allclose(next_state, state, rtol=1e-5, atol=1e-6):
        raise ValueError(f"steady state is not a fixed point of transition under a zero shock: {next_state}")

    initial = jnp.asarray(model.initial_state(init_rng))
    _match(initial, state_ss, "initial_state")
    _finite(initial, "initial_state")

    if _callable(model, "utility"):
        _utility_at(model, state, control)


def check_algorithm(model, algorithm):
    """Raise ``ValueError`` when ``model`` cannot be solved by ``algorithm``.

    ``algorithm`` is ``"apg"``, ``"deqn"``, or ``"time_iteration"``.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"algorithm must be one of {_ALGORITHMS}")
    check_model(model)
    if algorithm == "apg":
        _check_apg(model)
    else:
        _check_residuals(model)
        if algorithm == "time_iteration":
            _check_time_iteration(model)


def _check_apg(model):
    if not _callable(model, "utility"):
        raise ValueError("apg requires utility")
    steady = model.steady_state()
    state = jnp.asarray(steady.state)
    control = jnp.asarray(steady.control)
    shock = jnp.zeros_like(model.sample_shock(random.PRNGKey(0)))

    utility_grad = jax.grad(lambda current: model.utility(state, current))(control)
    _finite(utility_grad, "derivative of utility with respect to the control")
    transition_jac = jax.jacobian(lambda current: model.transition(state, current, shock))(control)
    _finite(transition_jac, "derivative of transition with respect to the control")
    _require(bool(jnp.any(transition_jac != 0)), "transition does not depend on the control")


def _check_residuals(model):
    if not (_callable(model, "expectation") and _callable(model, "residuals")):
        raise ValueError("deqn and time iteration require expectation and residuals")
    steady = model.steady_state()
    state = jnp.asarray(steady.state)
    control = jnp.asarray(steady.control)
    expectation = jnp.asarray(model.expectation(state, control))
    _require(expectation.ndim >= 1 and expectation.shape[-1] > 0, "expectation must end with a non-empty continuation axis")
    _finite(expectation, "expectation")
    residual = jnp.asarray(model.residuals(state, control, expectation))
    _require(residual.ndim == 1 and residual.shape[0] > 0, "residuals must be a non-empty vector, one entry per equation")
    _finite(residual, "residuals")
    if not jnp.allclose(residual, 0.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"residuals at the steady state must be zero, got {residual}")


def _check_time_iteration(model):
    if not hasattr(model, "n_endogenous"):
        raise ValueError("time iteration requires n_endogenous")
    n_endogenous = int(model.n_endogenous)
    state_dim = int(jnp.asarray(model.state_ss).shape[0])
    shock_dim = int(jnp.asarray(model.sample_shock(random.PRNGKey(0))).shape[0])
    _require(n_endogenous == 1, "time iteration grids one endogenous state")
    _require(
        state_dim - n_endogenous == shock_dim,
        "the exogenous part of the state must have the same dimension as the shock",
    )


def _scales(model):
    state_ss = jnp.asarray(model.state_ss)
    state_sd = jnp.asarray(model.state_sd)
    control_ss = jnp.asarray(model.control_ss)
    control_sd = jnp.asarray(model.control_sd)
    for name, value in (
        ("state_ss", state_ss),
        ("state_sd", state_sd),
        ("control_ss", control_ss),
        ("control_sd", control_sd),
    ):
        _require(value.ndim == 1 and value.shape[0] > 0, f"{name} must be a non-empty vector")
        _finite(value, name)
    _require(state_ss.shape == state_sd.shape, "state_ss and state_sd must share a shape")
    _require(control_ss.shape == control_sd.shape, "control_ss and control_sd must share a shape")
    _require(bool(jnp.all(state_sd > 0)), "state_sd must be positive")
    _require(bool(jnp.all(control_sd > 0)), "control_sd must be positive")
    return state_ss, state_sd, control_ss, control_sd


def _utility_at(model, state, control):
    value = jnp.asarray(model.utility(state, control))
    _require(value.ndim == 0, "utility must be a scalar")
    _finite(value, "utility")


def _callable(model, name):
    return callable(getattr(model, name, None))


def _match(value, reference, name):
    _require(value.shape == reference.shape, f"{name} has shape {value.shape}, expected {reference.shape}")
    _finite(value, name)


def _finite(value, name):
    _require(bool(jnp.all(jnp.isfinite(value))), f"{name} must be finite")


def _finite_tree(tree, name):
    leaves = jax.tree_util.tree_leaves(tree)
    _require(len(leaves) > 0, f"{name} must contain parameters")
    for leaf in leaves:
        _finite(jnp.asarray(leaf), name)


def _require(condition, message):
    if not condition:
        raise ValueError(message)
