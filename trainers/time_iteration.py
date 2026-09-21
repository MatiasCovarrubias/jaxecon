"""Time iteration on the shared residual, using the existing Newton step."""

import jax
from jax import numpy as jnp


def train_time_iteration(model, config):
    from TimeIteration.algorithm.interpolation import interp1d
    from TimeIteration.algorithm.newton import solve_all_points
    from TimeIteration.models.processes import rouwenhorst

    n_a = int(config["n_a"])
    n_k = int(config["n_k"])
    width = jnp.asarray(config["control_width"], dtype=model.state_ss.dtype)
    a_nodes, markov = rouwenhorst(n_a, model.params.rho, model.params.shock_sd)
    logk_ss = model.state_ss[0]
    logk_grid = logk_ss + jnp.linspace(-width, width, n_k, dtype=model.state_ss.dtype)
    control = jnp.broadcast_to(model.control_ss, (n_a, n_k, model.control_ss.shape[0]))

    def clamp(saving_rate):
        return jnp.clip(saving_rate, model.control_ss - width, model.control_ss + width)

    def point_residual(a_index, logk, saving_rate, policy):
        saving_rate = clamp(saving_rate)
        state = jnp.stack([logk, a_nodes[a_index]])
        shock = jnp.zeros((1,), dtype=logk.dtype)
        logk_next = model.transition(state, saving_rate, shock)[0]

        def at_node(a_next, saving_row):
            next_saving = interp1d(logk_next, logk_grid, saving_row)
            next_state = jnp.stack([logk_next, a_next])
            return model.expectation(next_state, clamp(next_saving[None]))[0]

        continuation = jnp.dot(markov[a_index], jax.vmap(at_node)(a_nodes, policy[..., 0]))
        return model.residuals(state, saving_rate, continuation[None])

    def update(policy):
        a_index = jnp.arange(n_a)

        def residual_at(i, logk, saving_rate):
            return point_residual(i, logk, saving_rate, policy)

        updated = solve_all_points(residual_at, a_index, logk_grid, policy, int(config["newton_steps"]))
        return clamp(updated)

    for _ in range(int(config["ti_iterations"])):
        control = update(control)

    def grid_residual(a_index, logk, saving_rate):
        return point_residual(a_index, logk, saving_rate, control)

    rows = jax.vmap(
        lambda i, saving_row: jax.vmap(lambda logk, saving_rate: grid_residual(i, logk, saving_rate))(logk_grid, saving_row)
    )(jnp.arange(n_a), control)
    return {
        "residual_norm": float(jnp.max(jnp.abs(rows))),
        "iterations": float(config["ti_iterations"]),
    }
