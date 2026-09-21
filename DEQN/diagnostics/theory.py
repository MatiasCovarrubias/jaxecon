"""Single-state diagnostics for DEQN residual and Monte Carlo mechanics."""

from collections.abc import Callable
from typing import Any

import jax
from jax import lax
from jax import numpy as jnp


def _resolve_policy_fn(
    params: Any,
    apply_fn: Callable | None,
    policy_fn: Callable | None,
) -> Callable:
    if policy_fn is not None and (params is not None or apply_fn is not None):
        raise ValueError("provide either policy_fn or params/apply_fn")
    if policy_fn is not None:
        return policy_fn
    if params is None or apply_fn is None:
        raise ValueError("provide policy_fn or both params and apply_fn")
    return lambda state: apply_fn(params, state)


def _evaluate(
    model,
    state,
    rng,
    mc_draws,
    policy_fn,
    residual_kind,
    stop_gradient,
):
    policy = policy_fn(state)
    shocks = model.mc_shocks(rng, mc_draws)
    next_states = jax.vmap(
        model.step, in_axes=(None, None, 0)
    )(state, policy, shocks)
    next_policies = jax.vmap(policy_fn)(next_states)
    per_draw_expectation = jax.vmap(model.expect_realization)(
        next_states, next_policies
    )
    expected_return = jnp.mean(per_draw_expectation, axis=0)
    expected_for_residual = (
        lax.stop_gradient(expected_return)
        if stop_gradient
        else expected_return
    )
    capital_price = model.capital_price(state, policy)
    ratio_residual = model.euler_residual(
        state, policy, expected_for_residual
    )
    levels_residual = (
        capital_price - model.beta * expected_for_residual
    )
    per_draw_ratio_residual = jax.vmap(
        lambda expected: model.euler_residual(state, policy, expected)
    )(per_draw_expectation)
    per_draw_levels_residual = (
        capital_price - model.beta * per_draw_expectation
    )
    mapping = model.action_euler_mapping(state, policy)
    action_euler_residual = -mapping.M * levels_residual
    per_draw_action_euler_residual = (
        -mapping.M * per_draw_levels_residual
    )
    residual = (
        ratio_residual if residual_kind == "ratio" else levels_residual
    )
    loss = jnp.mean(jnp.square(residual))
    return {
        "policy": policy,
        "shocks": shocks,
        "next_states": next_states,
        "next_policies": next_policies,
        "per_draw_expectation": per_draw_expectation,
        "expected_return": expected_return,
        "expected_return_for_residual": expected_for_residual,
        "capital_price": capital_price,
        "per_draw_ratio_residual": per_draw_ratio_residual,
        "per_draw_levels_residual": per_draw_levels_residual,
        "action_euler_mapping": mapping.M,
        "cap_binding": mapping.cap_binding,
        "per_draw_action_euler_residual": per_draw_action_euler_residual,
        "action_euler_residual": action_euler_residual,
        "ratio_residual": ratio_residual,
        "levels_residual": levels_residual,
        "residual": residual,
        "loss": loss,
    }


def evaluate_single_state(
    model,
    state,
    rng,
    mc_draws,
    *,
    params: Any = None,
    apply_fn: Callable | None = None,
    policy_fn: Callable | None = None,
    residual_kind: str = "ratio",
    stop_gradient: bool = True,
    return_gradient: bool = False,
) -> dict[str, Any]:
    """Evaluate one DEQN state with explicit inner Monte Carlo mechanics.

    ``policy_fn`` must map one state to one policy.  Alternatively, provide
    ``params`` and an ``apply_fn`` with the usual ``apply_fn(params, state)``
    signature.  The returned arrays retain the draw dimension in
    ``per_draw_expectation`` and the two ``per_draw_*_residual`` entries.

    The ratio residual preserves the DEQN convention,
    ``capital_price / (beta * expected_return) - 1`` (or a model's
    ``euler_residual`` implementation), where ``expected_return`` is the
    sample mean over inner draws.  Because the reciprocal is nonlinear, a
    one-draw ratio residual is not silently treated as an unbiased estimator.

    The levels residual is the affine alternative
    ``current marginal-capital-price - beta * expected_return``.  With
    ``stop_gradient=True``, only the value used as the continuation
    expectation is stopped; the current policy and current marginal
    capital price remain differentiable.  ``loss`` is the mean squared
    selected residual.  Set ``return_gradient=True`` to also return its
    gradient with respect to ``params``; this requires ``params`` and
    ``apply_fn``. For the smooth saving-rate RBC, ``action_euler_residual``
    converts the capital-price residual to network-action coordinates as
    ``e_z = -M * levels_residual``.

    Args:
        model: DEQN economic model implementing the RBC model protocol.
        state: One normalized state, without a leading batch dimension.
        rng: JAX PRNG key used for the inner shock draws.
        mc_draws: Number of inner Monte Carlo draws.
        params: Parameters passed to ``apply_fn``.
        apply_fn: Policy application function accepting ``(params, state)``.
        policy_fn: Direct policy function accepting ``state``.
        residual_kind: Either ``"ratio"`` or ``"levels"``.
        stop_gradient: Whether to stop the gradient through the sample
            continuation expectation.
        return_gradient: Whether to compute a parameter gradient in addition
            to the scalar loss.

    Returns:
        A dictionary containing per-draw states, policies, expectations, both
        residual conventions, the selected residual, and its scalar loss.
        ``gradient`` is included when requested.
    """
    if residual_kind not in {"ratio", "levels"}:
        raise ValueError("residual_kind must be 'ratio' or 'levels'")
    if mc_draws < 1:
        raise ValueError("mc_draws must be positive")
    if return_gradient and (params is None or apply_fn is None):
        raise ValueError(
            "return_gradient requires params and apply_fn"
        )

    resolved_policy_fn = _resolve_policy_fn(params, apply_fn, policy_fn)
    quantities = _evaluate(
        model,
        state,
        rng,
        mc_draws,
        resolved_policy_fn,
        residual_kind,
        stop_gradient,
    )

    if return_gradient:
        def loss_for_params(parameter_values):
            parameter_policy_fn = lambda obs: apply_fn(
                parameter_values, obs
            )
            return _evaluate(
                model,
                state,
                rng,
                mc_draws,
                parameter_policy_fn,
                residual_kind,
                stop_gradient,
            )["loss"]

        quantities["loss"], quantities["gradient"] = jax.value_and_grad(
            loss_for_params
        )(params)
    return quantities


def finite_depth_policy_residual(
    euler_residual,
    envelope_operator,
    discount_rate,
    depth,
):
    """Propagate an action-coordinate Euler residual for a finite depth.

    ``envelope_operator[t]`` maps the policy residual at date ``t + 1``
    into the action coordinates at date ``t``.  The final date has no
    continuation term, so propagation is zero-padded at the right edge.
    A depth of zero returns the direct Euler residual.
    """
    if int(depth) != depth or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    residual = jnp.asarray(euler_residual)
    operator = jnp.asarray(envelope_operator)
    if residual.ndim < 1 or operator.shape[0] != residual.shape[0]:
        raise ValueError(
            "euler_residual and envelope_operator must share a time dimension"
        )

    def transported(next_residual):
        if operator.shape == residual.shape:
            scale = operator
            return scale * next_residual
        if operator.shape == residual.shape[:-1]:
            scale = operator if residual.ndim == 1 else operator[..., None]
            return scale * next_residual
        return jnp.einsum("tji,tj->ti", operator, next_residual)

    current = residual
    zero = jnp.zeros_like(current[:1])
    for _ in range(int(depth)):
        shifted = jnp.concatenate((current[1:], zero), axis=0)
        current = residual + discount_rate * transported(shifted)
    return current
