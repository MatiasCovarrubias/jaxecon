"""Layer 5: simulation and diagnostics. Euler errors reuse `arbitrage`."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax, random

from TimeIteration.algorithm.expectation import expectation
from TimeIteration.algorithm.interpolation import interp_policy
from TimeIteration.algorithm.solve import residual_on_grids
from TimeIteration.models.protocol import Grids
from TimeIteration.models.rbc import period_utility


class Trajectory(NamedTuple):
    K: Array
    a: Array
    C: Array
    Y: Array
    I: Array
    mu: Array
    s: Array


class ErgodicStats(NamedTuple):
    K_rel: Array
    I_rel: Array
    C_rel: Array
    s_mean: Array
    bind_frac: Array
    std_log_C: Array
    std_log_I: Array
    std_log_Y: Array


def allocation_on_grid(model: Any, params: Any, grids: Grids, x_grid: Array):
    def row(a, x_row):
        return jax.vmap(lambda K, x: model.auxiliary(params, a, K, x, grids.ss))(grids.K_grid, x_row)

    return jax.vmap(row)(grids.a_nodes, x_grid)


def simulate(model: Any, params: Any, grids: Grids, x_grid: Array, key: Array, T: int, N: int) -> Trajectory:
    cdf = jnp.cumsum(grids.P, axis=-1)
    n_a = grids.a_nodes.shape[0]
    a0 = jnp.argmin(jnp.abs(grids.a_nodes)).astype(jnp.int32)
    K0 = grids.ss.K

    def one_path(path_key):
        keys = random.split(path_key, T)

        def body(carry, step_key):
            a_idx, K = carry
            a = grids.a_nodes[a_idx]
            x = interp_policy(grids.K_grid, x_grid[a_idx], K)
            aux = model.auxiliary(params, a, K, x, grids.ss)
            K_next = model.transition(params, a, K, x)
            draw = random.uniform(step_key, dtype=K.dtype)
            a_next = jnp.clip(jnp.searchsorted(cdf[a_idx], draw, side="right"), 0, n_a - 1).astype(jnp.int32)
            return (a_next, K_next), (K, a, aux.C, aux.Y, aux.I, aux.mu, aux.s)

        _, traj = lax.scan(body, (a0, K0), keys)
        return Trajectory(*traj)

    return jax.vmap(one_path)(random.split(key, N))


def euler_errors(model: Any, params: Any, grids: Grids, x_grid: Array, a_idx: Array, K: Array) -> Array:
    def one(i, K_i):
        a = grids.a_nodes[i]
        x = interp_policy(grids.K_grid, x_grid[i], K_i)
        K_next = model.transition(params, a, K_i, x)
        Ex = expectation(
            grids.P[i],
            grids.a_nodes,
            grids.K_grid,
            x_grid,
            K_next,
            lambda a_next, K_next, x_next: model.expectand(params, a_next, K_next, x_next, grids.ss),
        )
        return model.arbitrage(params, a, K_i, x, Ex, grids.ss)

    return jax.vmap(one)(a_idx, K)


def euler_errors_on_grid(model: Any, params: Any, grids: Grids, x_grid: Array) -> Array:
    return residual_on_grids(model, params, grids, x_grid)


def welfare_ce(C: Array, C_base: Array, params: Any) -> Array:
    horizon = C.shape[-1]
    discount = params.beta ** jnp.arange(horizon)
    rental = 1.0 / params.beta - 1.0 + params.delta
    K_ss = (params.alpha / rental) ** (1.0 / (1.0 - params.alpha))
    C_ss = K_ss**params.alpha - params.delta * K_ss
    c_bar = params.cbar_frac * C_ss
    welfare = jnp.sum(period_utility(C, params.sigma, c_bar) * discount, axis=-1)
    baseline = jnp.sum(period_utility(C_base, params.sigma, c_bar) * discount, axis=-1)
    weight = jnp.sum(discount)
    log_ce = jnp.exp((welfare - baseline) / weight) - 1.0
    crra_ce = (welfare / baseline) ** (1.0 / (1.0 - params.sigma)) - 1.0
    return jnp.where(jnp.abs(params.sigma - 1.0) < 1e-12, log_ce, crra_ce)


def switching_boundary(model: Any, params: Any, grids: Grids, x_grid: Array, tol: float = 1e-8) -> Array:
    I = allocation_on_grid(model, params, grids, x_grid).I
    return I <= params.i_min_frac * grids.ss.I + tol


def ergodic_stats(params: Any, grids: Grids, traj: Trajectory) -> ErgodicStats:
    I_min = params.i_min_frac * grids.ss.I
    return ErgodicStats(
        K_rel=jnp.mean(traj.K) / grids.ss.K,
        I_rel=jnp.mean(traj.I) / grids.ss.I,
        C_rel=jnp.mean(traj.C) / grids.ss.C,
        s_mean=jnp.mean(traj.s),
        bind_frac=jnp.mean(traj.I <= I_min + 1e-8),
        std_log_C=jnp.mean(jnp.std(jnp.log(jnp.maximum(traj.C, 1e-12)), axis=-1)),
        std_log_I=jnp.mean(jnp.std(jnp.log(jnp.maximum(traj.I, 1e-12)), axis=-1)),
        std_log_Y=jnp.mean(jnp.std(jnp.log(jnp.maximum(traj.Y, 1e-12)), axis=-1)),
    )
