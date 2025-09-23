from typing import Sequence

import flax.linen as nn
import jax
from jax import numpy as jnp


class NeuralNet_compute_expects(nn.Module):
    features_policies: Sequence[int]
    features_expects: Sequence[int]
    dropout_rate: float = 0.0  # Default dropout rate
    layer_norm: bool = False  # Default layer norm
    precision = jnp.float32

    @nn.compact
    def __call__(self, x, freeze_policies: bool = False, freeze_expects: bool = True, deterministic: bool = False):

        policy = x
        for i, feat in enumerate(self.features_policies[:-1]):
            policy = nn.relu(nn.Dense(feat, param_dtype=self.precision, name=f"pol_ly_{i}")(policy))
            if self.dropout_rate > 0:
                policy = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(policy)
            if self.layer_norm:
                policy = nn.LayerNorm(dtype=self.precision)(policy)
        policy = nn.softplus(
            nn.Dense(self.features_policies[-1], param_dtype=self.precision, name="pol_output_ly")(policy)
        )
        if freeze_policies:
            policy = jax.lax.stop_gradient(policy)

        expect = x
        for i, feat in enumerate(self.features_expects[:-1]):
            expect = nn.relu(nn.Dense(feat, param_dtype=self.precision, name=f"exp_ly_{i}")(expect))
            if self.layer_norm:
                expect = nn.LayerNorm(dtype=self.precision)(expect)
        expect = nn.softplus(
            nn.Dense(self.features_expects[-1], param_dtype=self.precision, name="exp_output_ly")(expect)
        )
        if freeze_expects:
            expect = jax.lax.stop_gradient(expect)

        return policy, expect
