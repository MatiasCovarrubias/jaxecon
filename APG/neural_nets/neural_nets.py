from typing import Callable, Optional, Sequence

import jax
from flax import linen as nn
from jax import numpy as jnp

from DEQN.neural_nets.neural_nets import PolicyNet


class PolicyNetLoglinear(nn.Module):
    """PolicyNet residual around an in-house log-linear baseline.

    ``C`` maps unit-scale state deviations to action deviations, matching
    ``DEQN.neural_nets.with_loglinear_baseline.NeuralNet``. Residual weights
    start at zero, so the untrained network is the linear policy.
    """

    features: Sequence[int]
    n_out: int
    C: jnp.ndarray
    states_sd: jnp.ndarray
    policies_sd: jnp.ndarray
    precision: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x_2d = x.reshape(-1, x.shape[-1])
        state_logdev = x_2d * self.states_sd[None, :]
        baseline_logdev = state_logdev @ self.C.T
        baseline = baseline_logdev / self.policies_sd[None, :]
        h = x_2d
        for feat in self.features:
            h = nn.relu(nn.Dense(feat, param_dtype=self.precision)(h))
        residual = nn.Dense(
            self.n_out,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            param_dtype=self.precision,
        )(h)
        output = baseline + residual
        if x.ndim == 1:
            output = output.reshape(-1)
        return output


class ActorCritic(nn.Module):
    actions_dim: int
    hidden_dims_actor: Sequence[int]
    hidden_dims_critic: Sequence[int]
    precision: jnp.dtype = jnp.float32
    activations: Callable[[jax.Array], jax.Array] = nn.relu
    activation_final_critic: Optional[Callable[[jax.Array], jax.Array]] = None
    policy_output_bias_init: float | None = None
    loglinear_C: jnp.ndarray | None = None
    loglinear_states_sd: jnp.ndarray | None = None
    loglinear_policies_sd: jnp.ndarray | None = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        stop_critic_input_gradient: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        if self.loglinear_C is None:
            action = PolicyNet(
                features=self.hidden_dims_actor,
                n_out=self.actions_dim,
                precision=self.precision,
                output_bias_init=self.policy_output_bias_init,
                name="policy",
            )(x)
        else:
            action = PolicyNetLoglinear(
                features=self.hidden_dims_actor,
                n_out=self.actions_dim,
                C=self.loglinear_C,
                states_sd=self.loglinear_states_sd,
                policies_sd=self.loglinear_policies_sd,
                precision=self.precision,
                name="policy",
            )(x)

        value = jax.lax.stop_gradient(x) if stop_critic_input_gradient else x
        for size in self.hidden_dims_critic:
            value = self.activations(nn.Dense(size, param_dtype=self.precision)(value))
        value = nn.Dense(1, param_dtype=self.precision)(value)
        if self.activation_final_critic:
            value = self.activation_final_critic(value)
        value = jnp.squeeze(value, axis=-1)

        return action, value
