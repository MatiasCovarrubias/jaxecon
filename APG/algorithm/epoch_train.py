from jax import numpy as jnp, random, lax
import jax
from .loss import create_episode_loss_fn


def get_apg_train_fn(env, config):
    episode_loss_fn = create_episode_loss_fn(env, config)

    def episode_train_fn(train_state, epis_rng):
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, grads = grad_fn(train_state.params, train_state, epis_rng)
        grads = jax.lax.pmean(grads, axis_name="episodes")
        train_state = train_state.apply_gradients(grads=grads)
        grad_mean = jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree_map(jnp.mean, grads))))
        grad_max = jnp.max(jnp.array(jax.tree_util.tree_leaves(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))))
        grad_metrics = (grad_mean, grad_max)
        episode_metrics = (loss_metrics, grad_metrics)
        return train_state, episode_metrics

    def step_train_fn(train_state, step_rng):
        step_rng, *epis_rng = random.split(step_rng, config["epis_per_step"] + 1)
        train_state, batch_metrics = jax.vmap(
            episode_train_fn, in_axes=(None, 0), out_axes=(None, 0), axis_name="episodes"
        )(train_state, jnp.stack(epis_rng))
        return train_state, batch_metrics

    def epoch_train_fn(train_state, epoch_rng):
        """Vectorise and repeat the update to complete an epoch, made aout of steps_per_epoch episodes."""
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn
