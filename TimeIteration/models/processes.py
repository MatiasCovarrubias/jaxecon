"""Rouwenhorst discretization of a scalar AR(1)."""

import jax.numpy as jnp
from jax import Array


def rouwenhorst(n: int, rho: Array, sigma_eps: Array) -> tuple[Array, Array]:
    """Discretize `a' = rho * a + sigma_eps * ?`, `? ~ N(0, 1)`."""
    rho = jnp.asarray(rho)
    sigma_eps = jnp.asarray(sigma_eps)
    dtype = jnp.result_type(rho, sigma_eps)
    if n == 1:
        return jnp.zeros((1,), dtype=dtype), jnp.ones((1, 1), dtype=dtype)

    p = (1.0 + rho) / 2.0
    sigma_y = sigma_eps / jnp.sqrt(jnp.maximum(1.0 - rho**2, jnp.array(1e-18, dtype=dtype)))
    y = sigma_y * jnp.sqrt(jnp.array(n - 1, dtype=dtype)) * jnp.linspace(-1.0, 1.0, n, dtype=dtype)

    P = jnp.array([[p, 1.0 - p], [1.0 - p, p]], dtype=dtype)
    for m in range(2, n):
        P_old = P
        P = jnp.zeros((m + 1, m + 1), dtype=dtype)
        P = P.at[:m, :m].add(p * P_old)
        P = P.at[:m, 1:].add((1.0 - p) * P_old)
        P = P.at[1:, :m].add((1.0 - p) * P_old)
        P = P.at[1:, 1:].add(p * P_old)
        P = P.at[1:m, :].mul(0.5)
    P = P / jnp.sum(P, axis=1, keepdims=True)
    return y, P
