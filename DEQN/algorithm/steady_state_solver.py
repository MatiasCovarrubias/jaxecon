"""
Steady state solver for economic models using JAX and optimistix.

Finds (state_ss, policy_ss) such that:
1. State consistency: step(state_ss, policy_ss, 0) = state_ss
2. Equilibrium conditions: loss(state_ss, expect_ss, policy_ss) = 0

where expect_ss = expect_realization(state_ss, policy_ss) assumes
tomorrow's state and policy equal today's.
"""

from typing import NamedTuple

import jax.numpy as jnp
import optimistix as optx


class SteadyStateSolution(NamedTuple):
    """Result of steady state computation."""

    state: jnp.ndarray
    policy: jnp.ndarray
    loss_value: float
    state_residual_norm: float
    converged: bool


def solve_steady_state(
    econ_model,
    initial_state=None,
    initial_policy=None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    max_steps: int = 500,
) -> SteadyStateSolution:
    """
    Solve for the steady state of an economic model.

    Args:
        econ_model: Economic model with step, expect_realization, and loss methods
        initial_state: Initial guess for state (default: zeros)
        initial_policy: Initial guess for policy (default: zeros)
        atol: Absolute tolerance
        rtol: Relative tolerance
        max_steps: Maximum iterations

    Returns:
        SteadyStateSolution with state, policy, and convergence info
    """
    n_states = econ_model.dim_states
    n_policies = econ_model.dim_policies
    n_sectors = econ_model.n_sectors

    # Default to zeros (log-deviation steady state)
    if initial_state is None:
        initial_state = jnp.zeros(n_states)
    if initial_policy is None:
        initial_policy = jnp.zeros(n_policies)

    x0 = jnp.concatenate([initial_state, initial_policy])

    def loss_fn(x, args):
        """Combined loss: state consistency + equilibrium conditions."""
        state = x[:n_states]
        policy = x[n_states:]

        # State consistency: step(state, policy, 0) should equal state
        zero_shock = jnp.zeros(n_sectors)
        state_next = econ_model.step(state, policy, zero_shock)
        state_loss = jnp.mean((state_next - state) ** 2)

        # Equilibrium: expect uses same state/policy for "tomorrow"
        expect = econ_model.expect_realization(state, policy)
        eq_loss, _, _, _, _ = econ_model.loss(state, expect, policy)

        return state_loss + eq_loss

    solver = optx.BFGS(rtol=rtol, atol=atol)
    result = optx.minimise(loss_fn, solver, x0, args=None, max_steps=max_steps, throw=False)

    x_sol = result.value
    state_sol = x_sol[:n_states]
    policy_sol = x_sol[n_states:]

    # Compute final metrics
    zero_shock = jnp.zeros(n_sectors)
    state_residual = econ_model.step(state_sol, policy_sol, zero_shock) - state_sol
    expect = econ_model.expect_realization(state_sol, policy_sol)
    eq_loss, _, _, _, _ = econ_model.loss(state_sol, expect, policy_sol)

    return SteadyStateSolution(
        state=state_sol,
        policy=policy_sol,
        loss_value=float(eq_loss),
        state_residual_norm=float(jnp.linalg.norm(state_residual)),
        converged=bool(result.result == optx.RESULTS.successful),
    )
