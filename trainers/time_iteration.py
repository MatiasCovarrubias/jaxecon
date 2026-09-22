"""Time iteration on a log-capital grid, in the model's dtype.

The grid is uniform in ``log(K / K_ss)`` from ``log(k_min_rel)`` to
``log(k_max_rel)``. TFP nodes are the Rouwenhorst grid. Off-grid states are
linear in log capital, then in log TFP.
"""

import time

import jax
from jax import numpy as jnp

from trainers.policy import training_seconds


def interpolated_policy(logk_grid, a_nodes, saving, dtype):
    """Turn the solved grid into ``policy(state) -> saving rate``.

    ``saving`` has shape ``(n_a, n_k)`` on increasing ``logk_grid`` and
    ``a_nodes``. Outside the grid the same line extrapolates.
    """
    from TimeIteration.algorithm.interpolation import interp1d

    grid = jnp.asarray(logk_grid)
    productivity = jnp.asarray(a_nodes, dtype=grid.dtype)
    table = jnp.asarray(saving, dtype=grid.dtype)
    if table.ndim == 3:
        table = table[..., 0]

    def policy(state):
        state = jnp.asarray(state, dtype=grid.dtype)
        along_capital = jax.vmap(lambda row: interp1d(state[0], grid, row))(table)
        rate = interp1d(state[1], productivity, along_capital)
        return jnp.asarray(rate, dtype=dtype)[None]

    return policy


def train_time_iteration(model, config):
    from TimeIteration.algorithm.interpolation import interp1d
    from TimeIteration.algorithm.newton import solve_all_points
    from TimeIteration.models.processes import rouwenhorst

    dtype = model.state_ss.dtype
    n_a = int(config["n_a"])
    n_k = int(config["n_k"])
    a_nodes, markov = rouwenhorst(n_a, model.params.rho, model.params.shock_sd)
    ratio = jnp.geomspace(
        jnp.asarray(config["k_min_rel"], dtype=dtype),
        jnp.asarray(config["k_max_rel"], dtype=dtype),
        n_k,
    )
    logk_grid = model.state_ss[0] + jnp.log(ratio)
    control = jnp.broadcast_to(model.control_ss, (n_a, n_k, model.control_ss.shape[0]))

    def point_residual(a_index, logk, saving, policy):
        state = jnp.stack([logk, a_nodes[a_index]])
        shock = jnp.zeros((1,), dtype=dtype)
        logk_next = model.transition(state, saving, shock)[0]

        def at_node(a_next, saving_row):
            next_saving = interp1d(logk_next, logk_grid, saving_row)
            next_state = jnp.stack([logk_next, a_next])
            return model.expectation(next_state, next_saving[None])[0]

        continuation = jnp.dot(markov[a_index], jax.vmap(at_node)(a_nodes, policy[..., 0]))
        return model.residuals(state, saving, continuation[None])

    def update(policy):
        a_index = jnp.arange(n_a)

        def residual_at(i, logk, saving):
            return point_residual(i, logk, saving, policy)

        return solve_all_points(residual_at, a_index, logk_grid, policy, int(config["newton_steps"]))

    def residual_norm(policy):
        def grid_residual(a_index, logk, saving):
            return point_residual(a_index, logk, saving, policy)

        rows = jax.vmap(
            lambda i, saving_row: jax.vmap(lambda logk, saving: grid_residual(i, logk, saving))(logk_grid, saving_row)
        )(jnp.arange(n_a), policy)
        return jnp.max(jnp.abs(rows))

    update = jax.jit(update)
    residual_norm = jax.jit(residual_norm)
    tol = float(config["ti_tol"])
    iterations = 0
    first = 0.0
    rest = 0.0
    for _ in range(int(config["ti_iterations"])):
        tick = time.perf_counter()
        updated = update(control)
        error = jax.block_until_ready(jnp.max(jnp.abs(updated - control)))
        elapsed = time.perf_counter() - tick
        control = updated
        iterations += 1
        if iterations == 1:
            first = elapsed
        else:
            rest += elapsed
        if float(error) < tol:
            break
    compile_seconds, run_seconds = training_seconds(first, rest, iterations)

    saving = control[..., 0]
    policy = interpolated_policy(logk_grid, a_nodes, saving, dtype)

    return {
        "residual_norm": float(residual_norm(control)),
        "iterations": float(iterations),
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
    }, policy, {"logk_grid": logk_grid, "a_nodes": a_nodes, "saving": saving}
