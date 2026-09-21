"""Layer 1: 1-D policy interpolation. Exogenous states stay discrete."""

import jax.numpy as jnp
from jax import Array, vmap


def interp1d(x: Array, xp: Array, fp: Array) -> Array:
    """Linear interpolation with endpoint extrapolation."""
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    width = x1 - x0
    weight = (x - x0) / jnp.where(width == 0, jnp.ones_like(width), width)
    return fp[i] + (fp[i + 1] - fp[i]) * weight


def interp_policy(K_grid: Array, x_at_a: Array, K_query: Array) -> Array:
    """Interpolate each component of `x` at `K_query` for one exogenous node."""
    return vmap(lambda col: interp1d(K_query, K_grid, col))(x_at_a.T)
