from jax import numpy as jnp, lax, random
import jax
from simulation import create_episode_simul_fn
from loss import create_batch_loss_fn

def get_eval_fn(env, config):
  config = config["config_eval"]
  episode_simul_fn = create_episode_simul_fn(env, config)
  batch_loss_fn = create_batch_loss_fn(env, config)

  def episode_eval_fn(train_state, epis_rng):
    epis_rng, loss_rng = random.split(epis_rng, 2)
    epis_obs = episode_simul_fn(train_state, epis_rng)
    _, epis_metrics = batch_loss_fn(train_state.params, train_state, epis_obs, loss_rng)
    return epis_metrics

  def eval_fn(train_state, step_rng):
    epis_rng = random.split(step_rng, config["eval_n_epis"])
    losses, mean_accuracies, min_accuracies, mean_accs_focs, min_accs_focs = jax.vmap(episode_eval_fn, in_axes=(None,0))(train_state, jnp.stack(epis_rng))
    loss = jnp.mean(losses)
    mean_accuracy = jnp.mean(mean_accuracies)
    min_accuracy = jnp.min(min_accuracies)
    mean_accs_focs = jnp.mean(mean_accs_focs, axis=0)
    min_accs_focs = jnp.min(min_accs_focs, axis=0)
    return loss, mean_accuracy, min_accuracy, mean_accs_focs, min_accs_focs

  return eval_fn