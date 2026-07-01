"""
Evaluation functions for APG.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from flax.traverse_util import flatten_dict

from .loss import create_episode_loss_fn
from .simulation import create_episode_simul_fn


class ConvergenceMetrics(NamedTuple):
    loss: jax.Array
    actor_loss: jax.Array
    value_loss: jax.Array
    value_accuracy: jax.Array
    total_grad_norm: jax.Array
    actor_grad_norm: jax.Array
    critic_grad_norm: jax.Array
    total_grad_rms: jax.Array
    actor_grad_rms: jax.Array
    critic_grad_rms: jax.Array
    max_abs_grad: jax.Array


def _gradient_norm_metrics(grads, actor_dense_count):
    flat_grads = flatten_dict(grads)
    actor_sq = jnp.array(0.0)
    critic_sq = jnp.array(0.0)
    other_sq = jnp.array(0.0)
    actor_count = 0
    critic_count = 0
    other_count = 0
    max_abs_grad = jnp.array(0.0)

    for path, value in flat_grads.items():
        max_abs_grad = jnp.maximum(max_abs_grad, jnp.max(jnp.abs(value)))
        dense_name = next((str(part) for part in path if str(part).startswith("Dense_")), None)
        if dense_name is None:
            other_sq = other_sq + jnp.sum(jnp.square(value))
            other_count += value.size
            continue

        dense_idx = int(dense_name.split("_")[1])
        if dense_idx < actor_dense_count:
            actor_sq = actor_sq + jnp.sum(jnp.square(value))
            actor_count += value.size
        else:
            critic_sq = critic_sq + jnp.sum(jnp.square(value))
            critic_count += value.size

    total_sq = actor_sq + critic_sq + other_sq
    total_count = actor_count + critic_count + other_count
    return (
        jnp.sqrt(total_sq),
        jnp.sqrt(actor_sq),
        jnp.sqrt(critic_sq),
        jnp.sqrt(total_sq / max(total_count, 1)),
        jnp.sqrt(actor_sq / max(actor_count, 1)),
        jnp.sqrt(critic_sq / max(critic_count, 1)),
        max_abs_grad,
    )


def create_eval_fn(env, config):
    """Create evaluation function for APG.

    Args:
        env: Environment instance
        config: Configuration dictionary with eval parameters

    Returns:
        Function that evaluates the current policy and returns metrics
    """
    eval_periods = config.get("eval_periods_per_epis", config["periods_per_epis"])
    simul_episode = create_episode_simul_fn(env, eval_periods)

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
        return targets

    def episode_loss_fn(params, train_state, epis_rng):
        returns, trajectory, last_val = simul_episode(params, train_state, jnp.stack(epis_rng))
        values = trajectory.value
        targets = get_targets(trajectory, last_val)
        actor_loss = -returns
        value_loss = jnp.mean(jnp.square(values - targets))
        value_loss_perc = jnp.mean((values - targets) / targets)
        return actor_loss + value_loss, (actor_loss, value_loss, value_loss_perc)

    def episode_grads_and_metrics(train_state, epis_rng):
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, grads = grad_fn(train_state.params, train_state, epis_rng)
        grads = jax.lax.pmean(grads, axis_name="episodes")
        grad_mean = jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.mean, grads))))
        grad_max = jnp.max(
            jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads)))
        )
        grad_metrics = (grad_mean, grad_max)
        episode_metrics = (loss_metrics, grad_metrics)
        return episode_metrics

    def eval_fn(train_state, eval_rng):
        eval_n_epis = config.get("eval_n_epis", 1024)
        epis_rng = random.split(eval_rng, eval_n_epis)
        loss_metrics, grad_metrics = jax.vmap(
            episode_grads_and_metrics, in_axes=(None, 0), out_axes=0, axis_name="episodes"
        )(train_state, jnp.stack(epis_rng))
        eval_metrics = (
            jnp.mean(loss_metrics[0]),
            jnp.mean(loss_metrics[1][0]),
            jnp.mean(loss_metrics[1][1]),
            (1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100,
            jnp.mean(grad_metrics[0]),
            jnp.max(grad_metrics[1]),
        )
        return eval_metrics

    return eval_fn


def create_convergence_eval_fn(env, config):
    """Create a many-rollout gradient diagnostic for APG convergence."""
    diag_config = {
        **config,
        "periods_per_epis": config.get("diag_periods_per_epis", config["periods_per_epis"]),
    }
    diag_n_epis = config.get("diag_n_epis", config.get("eval_n_epis", 1024))
    actor_dense_count = config.get("actor_dense_count", len(config.get("layers_actor", [])) + 1)
    episode_loss_fn = create_episode_loss_fn(env, diag_config)

    def convergence_eval_fn(train_state, diag_rng):
        epis_rng = random.split(diag_rng, diag_n_epis)
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, episode_grads = jax.vmap(lambda key: grad_fn(train_state.params, train_state, key))(
            jnp.stack(epis_rng)
        )
        grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), episode_grads)
        grad_metrics = _gradient_norm_metrics(grads, actor_dense_count)

        loss = jnp.mean(loss_metrics[0])
        actor_loss = jnp.mean(loss_metrics[1][0])
        value_loss = jnp.mean(loss_metrics[1][1])
        value_accuracy = (1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100
        return ConvergenceMetrics(
            loss=loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            value_accuracy=value_accuracy,
            total_grad_norm=grad_metrics[0],
            actor_grad_norm=grad_metrics[1],
            critic_grad_norm=grad_metrics[2],
            total_grad_rms=grad_metrics[3],
            actor_grad_rms=grad_metrics[4],
            critic_grad_rms=grad_metrics[5],
            max_abs_grad=grad_metrics[6],
        )

    return convergence_eval_fn
