"""Linear-quadratic log-linear policy of an APG environment.

Ljungqvist and Sargent, chapter 5, applied to the planner problem

    max E sum_t beta^t U(x_t, a_t)   s.t.   x_{t+1} = G(x_t, a_t, e_{t+1}).

At the deterministic steady state the return is expanded to second order and
the transition to first order. Because ``G`` is nonlinear, the object that is
expanded is the Lagrangian period return ``U + beta * lambda . G`` with
``lambda`` the steady-state co-state; the second-order terms of ``G`` enter
through it. The result is a discounted LQ problem whose solution is the
discounted Riccati equation and a feedback ``action = C @ state``. ``C`` is the
analogue of Dynare's first-order policy matrix, computed from the environment
rather than loaded from data.

Notation follows Sargent: ``Q``, ``R``, ``W`` are the state, action, and cross
blocks of the Hessian of the Lagrangian return; ``A``, ``B``, ``D`` are the
Jacobians of ``G`` in state, action, and shock. Coordinates are those of the
environment (on the shared RBC, log deviations of ``(K, a)`` and the
saving-rate logit deviation), with unit scales.

The environment must expose ``state_sd``, ``policies_sd`` and
``set_scales(state_sd, policies_sd)`` in addition to the rollout interface.
"""

from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from scipy import linalg as sla


class LogLinearSolution(NamedTuple):
    """Linear policy ``action = C @ state`` in unit-scale environment coordinates."""

    C: np.ndarray
    P: np.ndarray
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    W: np.ndarray
    costate: np.ndarray
    foc_residual: np.ndarray
    states_sd: np.ndarray
    policies_sd: np.ndarray
    spectral_radius: float

    @property
    def A_closed(self):
        return self.A + self.B @ self.C


def _np64(x):
    return np.asarray(jax.device_get(x), dtype=np.float64)


def linearize_environment(env, rng=None):
    """Jacobians of ``transition`` and Hessian of the Lagrangian return at the steady state.

    Returns numpy float64 arrays ``A, B, D, Q, R, W, costate, foc_residual``.
    """
    rng = random.PRNGKey(0) if rng is None else rng
    state = jnp.asarray(env.initial_state(rng, 0, mode="box"))
    action = jnp.asarray(env.deterministic_steady_state_action())
    shock = jnp.zeros_like(jnp.asarray(env.sample_shock(rng)))
    n_s = state.shape[-1]
    beta = jnp.asarray(env.discount_rate, dtype=state.dtype)

    A = jax.jacobian(lambda s: env.transition(s, action, shock))(state)
    B = jax.jacobian(lambda a: env.transition(state, a, shock))(action)
    D = jax.jacobian(lambda z: env.transition(state, action, z))(shock)

    # Envelope condition lambda = U_x + beta A' lambda; FOC U_a + beta B' lambda = 0.
    U_x = jax.grad(lambda s: env.training_reward(s, action))(state)
    U_a = jax.grad(lambda a: env.training_reward(state, a))(action)
    costate = jnp.linalg.solve(jnp.eye(n_s, dtype=state.dtype) - beta * A.T, U_x)
    foc_residual = U_a + beta * (B.T @ costate)

    def lagrangian_return(z):
        s, a = z[:n_s], z[n_s:]
        return env.training_reward(s, a) + beta * jnp.dot(costate, env.transition(s, a, shock))

    hessian = jax.hessian(lagrangian_return)(jnp.concatenate([state, action]))
    hessian = 0.5 * (hessian + hessian.T)
    return {
        "A": _np64(A),
        "B": _np64(B),
        "D": _np64(D),
        "Q": _np64(hessian[:n_s, :n_s]),
        "R": _np64(hessian[n_s:, n_s:]),
        "W": _np64(hessian[:n_s, n_s:]),
        "costate": _np64(costate),
        "foc_residual": _np64(foc_residual),
    }


def _feedback(A, B, P, R, W, beta):
    return -np.linalg.solve(R + beta * (B.T @ P @ B), W.T + beta * (B.T @ P @ A))


def iterate_riccati(A, B, Q, R, W, beta, max_iter=20000, tol=1e-12):
    """Value-function iteration on the discounted LQ Bellman equation from ``P = 0``."""
    P = np.zeros_like(Q)
    for _ in range(max_iter):
        C = _feedback(A, B, P, R, W, beta)
        closed = A + B @ C
        P_new = Q + W @ C + C.T @ W.T + C.T @ R @ C + beta * (closed.T @ P @ closed)
        P_new = 0.5 * (P_new + P_new.T)
        if np.max(np.abs(P_new - P)) < tol:
            return P_new, C
        P = P_new
    raise RuntimeError(f"Riccati iteration did not converge in {max_iter} steps")


def solve_riccati(A, B, Q, R, W, beta, solver="scipy"):
    """Return ``(P, C)`` for the discounted LQ maximizer with value ``0.5 x' P x``.

    ``solver="scipy"`` calls the discrete algebraic Riccati solver on the
    equivalent minimization with ``sqrt(beta)``-scaled dynamics;
    ``solver="iterate"`` runs value-function iteration.
    """
    if np.any(np.linalg.eigvalsh(R) >= 0):
        raise ValueError("the return is not strictly concave in the action at the steady state")
    if solver == "iterate":
        return iterate_riccati(A, B, Q, R, W, beta)
    if solver != "scipy":
        raise ValueError("solver must be 'scipy' or 'iterate'")
    root = np.sqrt(beta)
    P = -sla.solve_discrete_are(root * A, root * B, -Q, -R, s=-W)
    return P, _feedback(A, B, P, R, W, beta)


def stationary_sd(A_closed, C, D, floor=1e-8):
    """Standard deviations of state and action under ``x' = A_closed x + D e``, ``e`` standard normal.

    Coordinates whose stationary variance is below ``floor`` get scale one, so
    that they can still be used as divisors.
    """
    covariance = sla.solve_discrete_lyapunov(A_closed, D @ D.T)
    covariance = 0.5 * (covariance + covariance.T)

    def to_sd(variance):
        sd = np.sqrt(np.maximum(variance, 0.0))
        return np.where(sd < floor, 1.0, sd)

    return to_sd(np.diag(covariance)), to_sd(np.diag(C @ covariance @ C.T))


def solve_loglinear(env, rng=None, solver="scipy"):
    """Compute the log-linear policy of ``env`` in unit-scale coordinates.

    The environment scales are set to one while differentiating, so that ``C``
    maps environment deviations to action deviations, and restored on exit.
    """
    state_sd0, policies_sd0 = env.state_sd, env.policies_sd
    env.set_scales(jnp.ones_like(state_sd0), jnp.ones_like(policies_sd0))
    try:
        lin = linearize_environment(env, rng)
    finally:
        env.set_scales(state_sd0, policies_sd0)

    beta = float(env.discount_rate)
    P, C = solve_riccati(lin["A"], lin["B"], lin["Q"], lin["R"], lin["W"], beta, solver=solver)
    A_closed = lin["A"] + lin["B"] @ C
    states_sd, policies_sd = stationary_sd(A_closed, C, lin["D"])
    return LogLinearSolution(
        C=C,
        P=P,
        states_sd=states_sd,
        policies_sd=policies_sd,
        spectral_radius=float(np.max(np.abs(np.linalg.eigvals(A_closed)))),
        **lin,
    )


def print_loglinear_solution(solution):
    print("Log-linear policy (LQ / Riccati, computed from the environment):")
    print("  C (action = C @ state) =")
    for row in np.atleast_2d(solution.C):
        print("   ", np.array2string(row, precision=4, suppress_small=True))
    print(f"  FOC residual at SS: {solution.foc_residual}")
    print(f"  states_sd:   {solution.states_sd}")
    print(f"  policies_sd: {solution.policies_sd}")
    print(f"  spectral radius of A+BC: {solution.spectral_radius:.4f}")
