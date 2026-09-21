"""Finite-horizon checks for the frozen-policy gradient identity."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class TheoryDiagnostic(NamedTuple):
    """Return and gradients produced by the theory diagnostic."""

    return_value: jax.Array
    autodiff_gradient: object
    forward_gradient: object
    reverse_gradient: object
    finite_difference_gradient: object


class StatewisePolicyGradient(NamedTuple):
    """Statewise action-value derivatives and their parameter pullback."""

    states: jax.Array
    actions: jax.Array
    action_value_gradients: jax.Array
    discounts: jax.Array
    parameter_gradient: object


def _flat_jacobian(function, argument):
    output = jnp.asarray(function(argument))
    jacobian = jax.jacobian(function)(argument)
    output_size = output.size
    input_size = jnp.asarray(argument).size
    return jnp.reshape(jacobian, (output_size, input_size))


def _rollout(initial_state, shocks, params, policy_fn, transition_fn):
    states = []
    actions = []
    state = initial_state
    for shock in shocks:
        action = policy_fn(params, state)
        states.append(state)
        actions.append(action)
        state = transition_fn(state, action, shock)
    return states, actions


def finite_horizon_return(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
    terminal_value_fn=None,
):
    """Evaluate the discounted return for a fixed initial state and shocks."""

    state = initial_state
    return_value = jnp.zeros((), dtype=jnp.asarray(state).dtype)
    discount = jnp.asarray(1.0, dtype=jnp.asarray(state).dtype)
    for shock in shocks:
        action = policy_fn(params, state)
        return_value = return_value + discount * reward_fn(state, action)
        state = transition_fn(state, action, shock)
        discount = discount * discount_rate
    if terminal_value_fn is not None:
        return_value = return_value + discount * terminal_value_fn(state)
    return return_value


def explicit_forward_gradient(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
):
    """Compute the parameter gradient with forward state sensitivities."""

    parameter_vector, unravel = ravel_pytree(params)
    parameter_size = parameter_vector.size
    state = initial_state
    state_size = jnp.size(jnp.asarray(state))
    sensitivity = jnp.zeros((state_size, parameter_size), dtype=parameter_vector.dtype)
    gradient = jnp.zeros((parameter_size,), dtype=parameter_vector.dtype)
    discount = jnp.asarray(1.0, dtype=parameter_vector.dtype)
    return_value = jnp.zeros((), dtype=jnp.asarray(state).dtype)

    for shock in shocks:
        current_params = unravel(parameter_vector)
        action = policy_fn(current_params, state)
        policy_state_jacobian = _flat_jacobian(
            lambda current_state: policy_fn(current_params, current_state),
            state,
        )
        policy_parameter_jacobian = _flat_jacobian(
            lambda current_vector: policy_fn(unravel(current_vector), state),
            parameter_vector,
        )
        reward_state_jacobian = jnp.ravel(
            jax.jacobian(lambda current_state: reward_fn(current_state, action))(state)
        )
        reward_action_jacobian = jnp.ravel(
            jax.jacobian(lambda current_action: reward_fn(state, current_action))(action)
        )
        transition_state_jacobian = _flat_jacobian(
            lambda current_state: transition_fn(current_state, action, shock),
            state,
        )
        transition_action_jacobian = _flat_jacobian(
            lambda current_action: transition_fn(state, current_action, shock),
            action,
        )

        action_sensitivity = policy_parameter_jacobian + policy_state_jacobian @ sensitivity
        gradient = gradient + discount * (
            reward_state_jacobian @ sensitivity
            + reward_action_jacobian @ action_sensitivity
        )
        sensitivity = (
            transition_state_jacobian @ sensitivity
            + transition_action_jacobian @ action_sensitivity
        )
        return_value = return_value + discount * reward_fn(state, action)
        state = transition_fn(state, action, shock)
        discount = discount * discount_rate

    return unravel(gradient)


def explicit_reverse_gradient(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
):
    """Compute the parameter gradient with reverse state adjoints."""

    return statewise_action_value_gradient(
        initial_state,
        shocks,
        params,
        policy_fn,
        reward_fn,
        transition_fn,
        discount_rate,
    ).parameter_gradient


def statewise_action_value_gradient(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
    terminal_value_fn=None,
):
    """Return pathwise ``q_a`` values and their policy-Jacobian pullback.

    The returned parameter gradient reconstructs the derivative of the
    sampled finite-horizon welfare, including
    ``discount_rate**T * terminal_value_fn(x_T)`` when a terminal value is
    supplied. State derivatives include the continuation delivered by the
    frozen policy, while parameter Jacobians hold each visited state fixed.
    """
    parameter_vector, unravel = ravel_pytree(params)
    states, actions = _rollout(
        initial_state, shocks, params, policy_fn, transition_fn
    )
    if states:
        final_state = transition_fn(
            states[-1], actions[-1], shocks[-1]
        )
    else:
        final_state = initial_state
    if terminal_value_fn is None:
        adjoint = jnp.zeros(
            (jnp.size(jnp.asarray(initial_state)),),
            dtype=parameter_vector.dtype,
        )
    else:
        adjoint = jnp.ravel(jax.grad(terminal_value_fn)(final_state))
    gradient = jnp.zeros((parameter_vector.size,), dtype=parameter_vector.dtype)
    discount = jnp.asarray(1.0, dtype=parameter_vector.dtype)
    discounts = [discount]
    for _ in shocks[1:]:
        discount = discount * discount_rate
        discounts.append(discount)

    action_value_gradients = []
    for period in reversed(range(len(states))):
        state = states[period]
        action = actions[period]
        shock = shocks[period]
        policy_state_jacobian = _flat_jacobian(
            lambda current_state: policy_fn(params, current_state),
            state,
        )
        policy_parameter_jacobian = _flat_jacobian(
            lambda current_vector: policy_fn(unravel(current_vector), state),
            parameter_vector,
        )
        reward_state_jacobian = jnp.ravel(
            jax.jacobian(lambda current_state: reward_fn(current_state, action))(state)
        )
        reward_action_jacobian = jnp.ravel(
            jax.jacobian(lambda current_action: reward_fn(state, current_action))(action)
        )
        transition_state_jacobian = _flat_jacobian(
            lambda current_state: transition_fn(current_state, action, shock),
            state,
        )
        transition_action_jacobian = _flat_jacobian(
            lambda current_action: transition_fn(state, current_action, shock),
            action,
        )

        action_value_gradient = (
            reward_action_jacobian
            + discount_rate * transition_action_jacobian.T @ adjoint
        )
        action_value_gradients.append(action_value_gradient)
        gradient = gradient + discounts[period] * (
            policy_parameter_jacobian.T @ action_value_gradient
        )
        adjoint = (
            reward_state_jacobian
            + policy_state_jacobian.T @ action_value_gradient
            + discount_rate * transition_state_jacobian.T @ adjoint
        )

    action_value_gradients.reverse()
    return StatewisePolicyGradient(
        states=jnp.stack(states),
        actions=jnp.stack(actions),
        action_value_gradients=jnp.stack(action_value_gradients),
        discounts=jnp.stack(discounts),
        parameter_gradient=unravel(gradient),
    )


def central_finite_difference_gradient(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
    epsilon=1e-3,
):
    """Approximate the parameter gradient with central finite differences."""

    parameter_vector, unravel = ravel_pytree(params)
    directions = jnp.eye(parameter_vector.size, dtype=parameter_vector.dtype)

    def evaluate(vector):
        return finite_horizon_return(
            initial_state,
            shocks,
            unravel(vector),
            policy_fn,
            reward_fn,
            transition_fn,
            discount_rate,
        )

    plus = jax.vmap(evaluate)(parameter_vector + epsilon * directions)
    minus = jax.vmap(evaluate)(parameter_vector - epsilon * directions)
    return unravel((plus - minus) / (2 * epsilon))


def policy_gradient_diagnostic(
    initial_state,
    shocks,
    params,
    policy_fn,
    reward_fn,
    transition_fn,
    discount_rate,
    epsilon=1e-3,
):
    """Compare autodiff, forward, reverse, and finite-difference gradients."""

    return_value = finite_horizon_return(
        initial_state,
        shocks,
        params,
        policy_fn,
        reward_fn,
        transition_fn,
        discount_rate,
    )
    autodiff_gradient = jax.grad(
        lambda current_params: finite_horizon_return(
            initial_state,
            shocks,
            current_params,
            policy_fn,
            reward_fn,
            transition_fn,
            discount_rate,
        )
    )(params)
    return TheoryDiagnostic(
        return_value=return_value,
        autodiff_gradient=autodiff_gradient,
        forward_gradient=explicit_forward_gradient(
            initial_state,
            shocks,
            params,
            policy_fn,
            reward_fn,
            transition_fn,
            discount_rate,
        ),
        reverse_gradient=explicit_reverse_gradient(
            initial_state,
            shocks,
            params,
            policy_fn,
            reward_fn,
            transition_fn,
            discount_rate,
        ),
        finite_difference_gradient=central_finite_difference_gradient(
            initial_state,
            shocks,
            params,
            policy_fn,
            reward_fn,
            transition_fn,
            discount_rate,
            epsilon,
        ),
    )
