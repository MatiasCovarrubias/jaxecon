from typing import Sequence

import flax.linen as nn
import jax
from jax import numpy as jnp


class NeuralNet(nn.Module):
    features: Sequence[int]
    precision: jnp.dtype  # Default precision

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, param_dtype=self.precision)(x))
        x = nn.softplus(nn.Dense(self.features[-1], param_dtype=self.precision)(x))
        return x


class NeuralNet_freezed_layers(nn.Module):
    features: Sequence[int]
    n_freeze_layers: int = 0  # Number of layers to freeze
    precision = jnp.float32  # Assuming 'precision' is defined elsewhere

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, param_dtype=self.precision)(x)
            if i == self.n_freeze_layers - 1:
                x = jax.lax.stop_gradient(nn.relu(x))
            else:
                x = nn.relu(x)

        # Last layer with softplus activation
        x = nn.softplus(nn.Dense(self.features[-1], param_dtype=self.precision)(x))
        return x


class NeuralNet_dropout(nn.Module):
    features: Sequence[int]
    n_freeze_layers: int = 2  # Default number of layers to freeze
    dropout_rate: float = 5.0  # Default dropout rate
    precision = jnp.float32  # Assuming 'precision' is defined elsewhere

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, param_dtype=self.precision)(x)
            x = nn.relu(x)
            if i == self.n_freeze_layers - 1:
                x = jax.lax.stop_gradient(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # Last layer with softplus activation
        x = nn.softplus(nn.Dense(self.features[-1], param_dtype=self.precision)(x))
        return x


def create_neural_net_builder(dim_policies: int, precision: jnp.dtype = jnp.float32):
    """
    Create a builder function for neural networks.

    This function returns a callable that can build neural networks,
    allowing you to pre-configure a builder and then call it to create networks.

    Args:
        dim_policies: Dimension of policy outputs
        precision: Numerical precision for parameters

    Returns:
        A callable that takes layers and returns a NeuralNet instance
    """

    def builder(layers: Sequence[int]) -> NeuralNet:
        """
        Build a neural network with the specified layer structure.

        Args:
            layers: List of hidden layer sizes

        Returns:
            Configured NeuralNet instance
        """
        return NeuralNet(
            features=list(layers) + [dim_policies],
            precision=precision,
        )

    return builder
