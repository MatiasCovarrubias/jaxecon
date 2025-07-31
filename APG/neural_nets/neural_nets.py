from jax import numpy as jnp
import jax
from flax import linen as nn
from typing import Sequence, Callable, Optional


class ActorCritic(nn.Module):
    actions_dim: int
    hidden_dims_actor: Sequence[int]
    hidden_dims_critic: Sequence[int]
    activations: Callable[[jax.Array], jax.Array] = nn.tanh
    activation_final_actor: Optional[Callable[[jax.Array], jax.Array]] = None
    activation_final_critic: Optional[Callable[[jax.Array], jax.Array]] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Actor Action
        action = x
        for size in self.hidden_dims_actor:
            action = self.activations(nn.Dense(size)(action))
        action = nn.Dense(self.actions_dim)(action)
        if self.activation_final_actor:
            action = self.activation_final_actor(action)

        # Critic Value
        value = x
        for size in self.hidden_dims_critic:
            value = self.activations(nn.Dense(size)(value))
        value = nn.Dense(1)(value)
        if self.activation_final_critic:
            value = self.activation_final_critic(value)
        value = jnp.squeeze(value, axis=-1)

        return action, value
