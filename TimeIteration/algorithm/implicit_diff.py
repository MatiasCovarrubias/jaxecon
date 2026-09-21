"""Layer 6: implicit differentiation of the time-iteration fixed point."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from TimeIteration.algorithm.solve import consistent_residual, solve
from TimeIteration.models.protocol import GridSpec


def implicit_vjp(model: Any, spec: GridSpec, x_star: Array, params: Any, cotangent: Array) -> Any:
    """`?x*/?p` VJP from `F(x*, p) = 0` via the implicit function theorem."""
    x_flat, unravel_x = ravel_pytree(x_star)
    g_flat, _ = ravel_pytree(cotangent)

    def F_x(x_vec):
        return ravel_pytree(consistent_residual(model, params, spec, unravel_x(x_vec)))[0]

    jac = jax.jacfwd(F_x)(x_flat)
    jac = jac + jnp.array(1e-14, dtype=x_flat.dtype) * jnp.eye(x_flat.size, dtype=x_flat.dtype)
    adjoint = jnp.linalg.solve(jac.T, g_flat)

    def F_p(p):
        return ravel_pytree(consistent_residual(model, p, spec, x_star))[0]

    _, vjp_p = jax.vjp(F_p, params)
    return jax.tree.map(lambda z: -z, vjp_p(adjoint)[0])


def implicit_policy_fn(model: Any, spec: GridSpec, x0: Array | None = None, **solve_opts):
    """Return `params -> x_grid` with IFT reverse-mode derivatives."""

    @partial(jax.custom_vjp)
    def solve_params(params):
        return solve(model, params, spec, x0=x0, **solve_opts).x_grid

    def fwd(params):
        x_grid = solve_params(params)
        return x_grid, (x_grid, params)

    def bwd(res, cotangent):
        x_grid, params = res
        return (implicit_vjp(model, spec, x_grid, params, cotangent),)

    solve_params.defvjp(fwd, bwd)
    return solve_params
