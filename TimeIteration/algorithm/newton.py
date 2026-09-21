"""Layer 3: damped Newton on a small dense residual, vmapped over the grid."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array, lax, vmap


def newton_solve(F: Callable[[Array], Array], x0: Array, n_steps: int = 15) -> Array:
    """Fixed-iteration Newton with one-step damping if the residual grows."""
    eye = jnp.eye(x0.shape[0], dtype=x0.dtype)

    def body(_, x):
        residual = F(x)
        jac = jax.jacfwd(F)(x) + jnp.array(1e-14, dtype=x0.dtype) * eye
        step = jnp.linalg.solve(jac, residual)
        x_full = x - step
        residual_full = F(x_full)
        worse = jnp.linalg.norm(residual_full) > jnp.linalg.norm(residual)
        return jnp.where(worse, x - 0.5 * step, x_full)

    return lax.fori_loop(0, n_steps, body, x0)


def solve_all_points(F_at, a_idx: Array, K_grid: Array, x_grid: Array, n_steps: int = 15) -> Array:
    """Newton-solve `F_at(a_idx, K, x) = 0` at every `(a, K)` node."""

    def solve_one(i, K, x0):
        return newton_solve(lambda x: F_at(i, K, x), x0, n_steps)

    def solve_row(i, x_row):
        return vmap(lambda K, x0: solve_one(i, K, x0))(K_grid, x_row)

    return vmap(solve_row)(a_idx, x_grid)
