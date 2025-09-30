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


def create_neural_net_loglinear_builder(
    C: jnp.ndarray, policies_sd: jnp.ndarray, dim_policies: int, param_dtype: jnp.dtype = jnp.float64
):
    """
    Create a builder function for log-linear baseline neural networks.

    This function returns a callable that can build log-linear neural networks
    with a baseline policy.

    Args:
        C: Linear coefficient matrix, shape (n_states, n_states)
        policies_sd: Policy standard deviations, shape (n_policies,)
        dim_policies: Dimension of policy outputs
        param_dtype: Numerical precision for parameters

    Returns:
        A callable that takes layers and returns a NeuralNet instance with log-linear baseline
    """

    def builder(layers: Sequence[int]) -> NeuralNet:
        """
        Build a log-linear neural network with the specified layer structure.

        Args:
            layers: List of hidden layer sizes

        Returns:
            Configured NeuralNet instance with log-linear baseline
        """
        return NeuralNet(
            features=list(layers) + [dim_policies],
            C=C,
            policies_sd=policies_sd,
            param_dtype=param_dtype,
        )

    return builder
