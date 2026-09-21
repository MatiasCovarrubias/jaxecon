import math
from typing import Callable, Sequence

import jax
from flax import linen as nn
from jax import numpy as jnp


def _softplus_with_gradient_floor(logits: jax.Array, floor: float) -> jax.Array:
    exact = jax.nn.softplus(logits)
    if floor == 0:
        return exact
    threshold = math.log(floor) - math.log1p(-floor)
    value_at_threshold = jax.nn.softplus(jnp.asarray(threshold, dtype=logits.dtype))
    carrier = jnp.where(
        logits >= threshold,
        exact,
        value_at_threshold + floor * (logits - threshold),
    )
    return jax.lax.stop_gradient(exact - carrier) + carrier


class MultiplierNet(nn.Module):
    """Positive state-dependent multiplier with an independent parameter tree."""

    features: Sequence[int]
    initial_value: float = 0.1
    gradient_floor: float = 0.0
    precision: jnp.dtype = jnp.float32
    activations: Callable[[jax.Array], jax.Array] = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        return_logits: bool = False,
    ) -> jax.Array:
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if not 0 <= self.gradient_floor < 1:
            raise ValueError("gradient_floor must satisfy 0 <= gradient_floor < 1")
        value = x
        for size in self.features:
            value = self.activations(nn.Dense(size, param_dtype=self.precision)(value))
        initial_logit = math.log(math.expm1(self.initial_value))
        logits = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(initial_logit),
            param_dtype=self.precision,
        )(value)
        if return_logits:
            return logits
        return _softplus_with_gradient_floor(logits, self.gradient_floor)


class GridMultiplierNet(nn.Module):
    """Positive state-dependent multiplier with local bilinear interpolation."""

    grid_size: int
    state_min: Sequence[float]
    state_max: Sequence[float]
    initial_value: float = 0.1
    gradient_floor: float = 0.0
    precision: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        return_logits: bool = False,
    ) -> jax.Array:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two")
        if len(self.state_min) != 2 or len(self.state_max) != 2:
            raise ValueError("grid multiplier requires two-dimensional bounds")
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if not 0 <= self.gradient_floor < 1:
            raise ValueError("gradient_floor must satisfy 0 <= gradient_floor < 1")

        initial_logit = math.log(math.expm1(self.initial_value))
        logits_grid = self.param(
            "logits",
            nn.initializers.constant(initial_logit),
            (self.grid_size, self.grid_size),
            self.precision,
        )
        state_min = jnp.asarray(self.state_min, dtype=self.precision)
        state_max = jnp.asarray(self.state_max, dtype=self.precision)
        coordinates = (x - state_min) / (state_max - state_min)
        coordinates = jnp.clip(coordinates, 0, 1) * (self.grid_size - 1)
        lower = jnp.floor(coordinates).astype(jnp.int32)
        lower = jnp.minimum(lower, self.grid_size - 2)
        upper = lower + 1
        weight = coordinates - lower

        lower_lower = logits_grid[lower[..., 0], lower[..., 1]]
        upper_lower = logits_grid[upper[..., 0], lower[..., 1]]
        lower_upper = logits_grid[lower[..., 0], upper[..., 1]]
        upper_upper = logits_grid[upper[..., 0], upper[..., 1]]
        logits = (
            (1 - weight[..., 0]) * (1 - weight[..., 1]) * lower_lower
            + weight[..., 0] * (1 - weight[..., 1]) * upper_lower
            + (1 - weight[..., 0]) * weight[..., 1] * lower_upper
            + weight[..., 0] * weight[..., 1] * upper_upper
        )[..., None]
        if return_logits:
            return logits
        return _softplus_with_gradient_floor(logits, self.gradient_floor)
