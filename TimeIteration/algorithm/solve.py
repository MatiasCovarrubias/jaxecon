"""Layer 4: time iteration. The model enters only through its residual."""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from TimeIteration.algorithm.expectation import expectation
from TimeIteration.algorithm.newton import solve_all_points
from TimeIteration.models.protocol import GridSpec, Grids


class SolveInfo(NamedTuple):
    iterations: Array
    error: Array
    residual_norm: Array
    converged: Array


class Solution(NamedTuple):
    x_grid: Array
    grids: Grids
    info: SolveInfo


def build_grids(model: Any, params: Any, spec: GridSpec) -> Grids:
    ss = model.steady_state(params)
    a_nodes, P = model.exog_process(params, spec)
    K_grid = model.endo_grid(params, ss, spec)
    return Grids(a_nodes=a_nodes, P=P, K_grid=K_grid, ss=ss)


def _point_residual(model, params, grids, x_grid, a_idx, K, x):
    a = grids.a_nodes[a_idx]
    K_next = model.transition(params, a, K, x)
    Ex = expectation(
        grids.P[a_idx],
        grids.a_nodes,
        grids.K_grid,
        x_grid,
        K_next,
        lambda a_next, K_next, x_next: model.expectand(params, a_next, K_next, x_next, grids.ss),
    )
    return model.arbitrage(params, a, K, x, Ex, grids.ss)


def residual_on_grids(model: Any, params: Any, grids: Grids, x_grid: Array) -> Array:
    a_idx = jnp.arange(grids.a_nodes.shape[0])

    def row(i, x_row):
        return jax.vmap(lambda K, x: _point_residual(model, params, grids, x_grid, i, K, x))(
            grids.K_grid, x_row
        )

    return jax.vmap(row)(a_idx, x_grid)


def consistent_residual(model: Any, params: Any, spec: GridSpec, x_grid: Array) -> Array:
    """Stacked arbitrage using the same policy for current `x` and expectations."""
    return residual_on_grids(model, params, build_grids(model, params, spec), x_grid)


def time_iteration_step(model: Any, params: Any, grids: Grids, x_grid: Array, newton_steps: int = 15) -> Array:
    a_idx = jnp.arange(grids.a_nodes.shape[0])

    def F_at(i, K, x):
        return _point_residual(model, params, grids, x_grid, i, K, x)

    return solve_all_points(F_at, a_idx, grids.K_grid, x_grid, newton_steps)


@partial(jax.jit, static_argnames=("model", "max_iter", "newton_steps"))
def _while_loop(model, params, grids, x0, tol, max_iter, damping, newton_steps):
    def cond(state):
        _, err, it = state
        return (err > tol) & (it < max_iter)

    def body(state):
        x, _, it = state
        x_new = time_iteration_step(model, params, grids, x, newton_steps)
        x_new = (1.0 - damping) * x_new + damping * x
        return x_new, jnp.max(jnp.abs(x_new - x)), it + 1

    return lax.while_loop(
        cond,
        body,
        (x0, jnp.asarray(jnp.inf, dtype=x0.dtype), jnp.asarray(0, dtype=jnp.int32)),
    )


def _anderson_loop(step_fn, x0, memory, tol, max_iter):
    x = x0
    hist_x = []
    hist_f = []
    error = jnp.asarray(jnp.inf, dtype=x0.dtype)
    for it in range(1, max_iter + 1):
        x_t = step_fn(x)
        residual = x_t - x
        error = jnp.max(jnp.abs(residual))
        if float(error) < float(tol):
            return x_t, jnp.asarray(it, dtype=jnp.int32), error
        hist_x.append(x)
        hist_f.append(residual)
        if len(hist_f) > memory:
            hist_x = hist_x[1:]
            hist_f = hist_f[1:]
        if len(hist_f) == 1:
            x = x_t
            continue
        F = jnp.stack([h.reshape(-1) for h in hist_f], axis=1)
        dF = F[:, 1:] - F[:, :-1]
        gamma, *_ = jnp.linalg.lstsq(dF, hist_f[-1].reshape(-1), rcond=None)
        T = jnp.stack([(hist_x[i] + hist_f[i]).reshape(-1) for i in range(len(hist_x))], axis=1)
        dT = T[:, 1:] - T[:, :-1]
        x = (x_t.reshape(-1) - dT @ gamma).reshape(x0.shape)
    return x, jnp.asarray(max_iter, dtype=jnp.int32), error


def solve(
    model: Any,
    params: Any,
    spec: GridSpec,
    x0: Array | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    damping: float = 0.0,
    newton_steps: int = 15,
    anderson_memory: int = 0,
) -> Solution:
    grids = build_grids(model, params, spec)
    if x0 is None:
        x0 = model.initial_policy(params, grids)
    tol = jnp.asarray(tol, dtype=x0.dtype)
    damping = jnp.asarray(damping, dtype=x0.dtype)
    if anderson_memory:
        step_fn = jax.jit(
            lambda x: (1.0 - damping) * time_iteration_step(model, params, grids, x, newton_steps) + damping * x
        )
        x, iterations, error = _anderson_loop(step_fn, x0, anderson_memory, tol, max_iter)
    else:
        x, error, iterations = _while_loop(model, params, grids, x0, tol, max_iter, damping, newton_steps)
    residual_norm = jnp.max(jnp.abs(residual_on_grids(model, params, grids, x)))
    info = SolveInfo(
        iterations=iterations,
        error=error,
        residual_norm=residual_norm,
        converged=error <= tol,
    )
    return Solution(x_grid=x, grids=grids, info=info)
