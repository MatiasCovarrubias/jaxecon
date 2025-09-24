from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class NeuralNet(nn.Module):
    features: Sequence[int]
    C: jnp.ndarray  # shape (n_states, n_states)
    policies_sd: jnp.ndarray  # shape (n_policies,)
    param_dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        # Ensure 2D for consistent slicing
        x_2d = x.reshape(-1, x.shape[-1])  # Always (batch, 111)

        # Baseline loglinear policy
        baseline = x @ self.C.T
        baseline = baseline * self.policies_sd[None, :]  # (batch, n_states)

        # Residual MLP
        h = x_2d
        for feat in self.features:
            h = nn.relu(nn.Dense(feat, param_dtype=self.param_dtype)(h))

        residual = nn.Dense(
            self.C.shape[0],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
        )(h)

        output = baseline + residual

        # Reshape output to match input shape structure
        if x.ndim == 1:
            output = output.reshape(-1)

        return output
