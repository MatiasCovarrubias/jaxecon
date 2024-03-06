from typing import Sequence
import flax.linen as nn
from jax import numpy as jnp
import jax

class NeuralNet(nn.Module):
  features: Sequence[int]
  precision = jnp.float32

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
    
class NeuralNet_compute_expects(nn.Module):
  features_policies: Sequence[int]
  features_expects: Sequence[int]
  precision = jnp.float32

  @nn.compact
  def __call__(self, x, freeze_policies: bool = False, freeze_expects: bool = True):
    policy = x
    for feat in self.features_policies[:-1]:
      policy = nn.relu(nn.Dense(feat, param_dtype=self.precision)(policy))
    policy = nn.softplus(nn.Dense(self.features_policies[-1], param_dtype=self.precision)(policy))

    expect = jax.lax.stop_gradient(x) if freeze_expects else x
    for feat in self.features_expects[:-1]:
      expect = nn.relu(nn.Dense(feat, param_dtype=self.precision)(expect))
    expect = nn.softplus(nn.Dense(self.features_expects[-1], param_dtype=self.precision)(expect))

    return policy, expect