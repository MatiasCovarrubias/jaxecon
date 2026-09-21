"""
Loss functions for APG.
"""

import jax
import jax.numpy as jnp

from .simulation import create_episode_simul_fn


def create_episode_loss_fn(env, config):
    """Create a loss function for a single episode.

    Args:
        env: Environment instance
        config: Configuration dictionary with rollout and value-target settings

    Returns:
        Function that computes episode loss and auxiliary metrics
    """
    use_terminal_value = config.get("use_terminal_value", False)
    simul_episode = create_episode_simul_fn(env, config, use_terminal_value=use_terminal_value)

    def get_targets(trajectory, last_val):
        """Compute GAE targets for value function updates."""

        def get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + env.discount_rate * next_value * (1 - done) - value
            gae = delta + env.discount_rate * config["gae_lambda"] * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectory,
            reverse=True,
            unroll=1,
        )
        targets = advantages + trajectory.value
        return jax.lax.stop_gradient(targets)

    def episode_loss_fn(params, train_state, epis_rng):
        returns, trajectory, last_val = simul_episode(params, train_state, epis_rng)
        actor_loss = -returns
        if not use_terminal_value:
            value_loss = jnp.zeros_like(actor_loss)
            value_loss_perc = jnp.full_like(actor_loss, jnp.nan)
            return actor_loss, (actor_loss, value_loss, value_loss_perc)

        values = trajectory.value
        targets = get_targets(trajectory, last_val)
        value_loss = jnp.mean(jnp.square(values - targets))
        value_loss_perc = jnp.mean((values - targets) / targets)
        return actor_loss + value_loss, (actor_loss, value_loss, value_loss_perc)

    return episode_loss_fn
