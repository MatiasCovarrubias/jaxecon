"""Layer 2: conditional expectation over the discretized exogenous chain."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, vmap

from TimeIteration.algorithm.interpolation import interp_policy


def expectation(
    P_row: Array,
    a_nodes: Array,
    K_grid: Array,
    x_grid: Array,
    K_next: Array,
    f: Callable[[Array, Array, Array], Array],
) -> Array:
    """`E[f(a', K_next, x'(a', K_next)) | a]` with `x'` interpolated in K."""

    def at_node(a, x_a):
        return f(a, K_next, interp_policy(K_grid, x_a, K_next))

    values = vmap(at_node)(a_nodes, x_grid)
    return jnp.einsum("j,j...->...", P_row, values)
