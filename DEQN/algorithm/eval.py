import jax
from jax import numpy as jnp
from jax import random


def create_eval_fn(config, episode_simul_fn, batch_loss_fn):

    config = config["config_eval"]

    def episode_eval_fn(train_state, epis_rng):
        epis_rng, loss_rng = random.split(epis_rng, 2)
        epis_obs = episode_simul_fn(train_state, epis_rng)
        obs_metrics = jnp.mean(epis_obs), jnp.max(epis_obs), jnp.mean(epis_obs[-1, :]), jnp.max(epis_obs[-1, :])
        _, loss_metrics = batch_loss_fn(train_state.params, train_state, epis_obs, loss_rng)
        return loss_metrics, obs_metrics

    def eval_fn(train_state, step_rng):
        epis_rng = random.split(step_rng, config["eval_n_epis"])
        loss_metrics, obs_metrics = jax.vmap(episode_eval_fn, in_axes=(None, 0))(train_state, jnp.stack(epis_rng))
        mean_losses, mean_accuracies, min_accuracies, mean_accs_focs, min_accs_focs = loss_metrics
        mean_loss = jnp.mean(mean_losses)
        mean_accuracy = jnp.mean(mean_accuracies)
        min_accuracy = jnp.min(min_accuracies)
        mean_accs_focs = jnp.mean(mean_accs_focs, axis=0)
        min_accs_focs = jnp.min(min_accs_focs, axis=0)
        mean_obsses, max_obsses, mean_obsses_terminal, max_obsses_terminal = obs_metrics
        mean_obs = jnp.mean(mean_obsses)
        max_obs = jnp.max(max_obsses)
        mean_obs_terminal = jnp.mean(mean_obsses_terminal)
        max_obs_terminal = jnp.max(max_obsses_terminal)
        return (
            mean_loss,
            mean_accuracy,
            min_accuracy,
            mean_accs_focs,
            min_accs_focs,
            mean_obs,
            max_obs,
            mean_obs_terminal,
            max_obs_terminal,
        )

    return eval_fn
