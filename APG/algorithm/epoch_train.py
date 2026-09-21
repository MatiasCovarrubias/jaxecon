"""Epoch training loop for APG."""

import jax
import jax.numpy as jnp
from jax import lax, random

from .loss import create_episode_loss_fn


def tree_grad_summary(grads):
    """Return (mean over all leaves' means, max absolute entry) of a gradient tree."""
    leaves = jax.tree_util.tree_leaves(grads)
    grad_mean = jnp.mean(jnp.array([jnp.mean(leaf) for leaf in leaves]))
    grad_max = jnp.max(jnp.array([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    return grad_mean, grad_max


def create_epoch_train_fn(env, config):
    """Create the epoch training function for APG.

    One step averages the episode gradient over ``epis_per_step`` episodes
    (``vmap`` + ``pmean``) and applies a single optimizer update. One epoch is
    ``steps_per_epoch`` such steps. Returns a function
    ``(train_state, rng) -> (train_state, rng, (loss_metrics, grad_metrics))``.
    """
    episode_loss_fn = create_episode_loss_fn(env, config)

    def episode_train_fn(train_state, epis_rng):
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, grads = grad_fn(train_state.params, train_state, epis_rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss_metrics, tree_grad_summary(grads))

    def step_train_fn(train_state, step_rng):
        epis_rng = random.split(step_rng, config["epis_per_step"])
        train_state, batch_metrics = jax.vmap(
            episode_train_fn, in_axes=(None, 0), out_axes=(None, 0), axis_name="batch"
        )(train_state, jnp.stack(epis_rng))
        return train_state, batch_metrics

    def epoch_train_fn(train_state, epoch_rng):
        """Vectorise and repeat the update to complete an epoch."""
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn
