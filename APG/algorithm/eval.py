"""Evaluation functions for APG."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from flax.traverse_util import flatten_dict

from .epoch_train import tree_grad_summary
from .loss import create_episode_loss_fn

ACTOR_MODULE_NAME = "policy"


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


def _gradient_norm_metrics(grads, has_critic):
    """Split gradient norms into actor and critic parts.

    With ``ActorCritic`` the actor lives under the ``policy`` sub-module and the
    critic's ``Dense_*`` layers sit at the top level. With a plain ``PolicyNet``
    every leaf is an actor leaf.
    """
    flat_grads = flatten_dict(grads)
    actor_sq = jnp.array(0.0)
    critic_sq = jnp.array(0.0)
    actor_count = 0
    critic_count = 0
    max_abs_grad = jnp.array(0.0)

    for path, value in flat_grads.items():
        max_abs_grad = jnp.maximum(max_abs_grad, jnp.max(jnp.abs(value)))
        is_actor = not has_critic or any(str(part) == ACTOR_MODULE_NAME for part in path)
        if is_actor:
            actor_sq = actor_sq + jnp.sum(jnp.square(value))
            actor_count += value.size
        else:
            critic_sq = critic_sq + jnp.sum(jnp.square(value))
            critic_count += value.size

    total_sq = actor_sq + critic_sq
    total_count = actor_count + critic_count
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
    """Create the per-epoch evaluation function.

    Runs ``eval_n_epis`` fresh rollouts of length ``eval_periods_per_epis`` with
    the same loss as training (no antithetic pairing) and returns
    ``(loss, actor_loss, value_loss, value_accuracy_pct, grad_mean, grad_max)``.
    """
    eval_config = {
        **config,
        "periods_per_epis": config.get("eval_periods_per_epis", config["periods_per_epis"]),
        "antithetic_episodes": False,
    }
    episode_loss_fn = create_episode_loss_fn(env, eval_config)

    def episode_grads_and_metrics(train_state, epis_rng):
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, grads = grad_fn(train_state.params, train_state, epis_rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        return loss_metrics, tree_grad_summary(grads)

    def eval_fn(train_state, eval_rng):
        eval_n_epis = config.get("eval_n_epis", 1024)
        epis_rng = random.split(eval_rng, eval_n_epis)
        loss_metrics, grad_metrics = jax.vmap(
            episode_grads_and_metrics, in_axes=(None, 0), out_axes=0, axis_name="batch"
        )(train_state, jnp.stack(epis_rng))
        return (
            jnp.mean(loss_metrics[0]),
            jnp.mean(loss_metrics[1][0]),
            jnp.mean(loss_metrics[1][1]),
            (1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100,
            jnp.mean(grad_metrics[0]),
            jnp.max(grad_metrics[1]),
        )

    return eval_fn


def create_convergence_eval_fn(env, config):
    """Create a many-rollout gradient diagnostic for APG convergence.

    Averages the episode gradient over ``diag_n_epis`` rollouts of length
    ``diag_periods_per_epis`` and reports actor, critic, and total gradient norms.
    """
    diag_config = {
        **config,
        "periods_per_epis": config.get("diag_periods_per_epis", config["periods_per_epis"]),
        "antithetic_episodes": False,
    }
    diag_n_epis = config.get("diag_n_epis", config.get("eval_n_epis", 1024))
    has_critic = bool(config.get("use_terminal_value", False))
    episode_loss_fn = create_episode_loss_fn(env, diag_config)

    def convergence_eval_fn(train_state, diag_rng):
        epis_rng = random.split(diag_rng, diag_n_epis)
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, episode_grads = jax.vmap(lambda key: grad_fn(train_state.params, train_state, key))(
            jnp.stack(epis_rng)
        )
        grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), episode_grads)
        grad_metrics = _gradient_norm_metrics(grads, has_critic)

        return ConvergenceMetrics(
            loss=jnp.mean(loss_metrics[0]),
            actor_loss=jnp.mean(loss_metrics[1][0]),
            value_loss=jnp.mean(loss_metrics[1][1]),
            value_accuracy=(1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100,
            total_grad_norm=grad_metrics[0],
            actor_grad_norm=grad_metrics[1],
            critic_grad_norm=grad_metrics[2],
            total_grad_rms=grad_metrics[3],
            actor_grad_rms=grad_metrics[4],
            critic_grad_rms=grad_metrics[5],
            max_abs_grad=grad_metrics[6],
        )

    return convergence_eval_fn
