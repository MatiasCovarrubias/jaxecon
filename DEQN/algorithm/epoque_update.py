from jax import numpy as jnp, lax, random
import jax

def get_epoch_train_fn(econ_model, config, episode_simul_fn, batch_loss_fn):

  def batch_train_fn(train_state, batch_obs, loss_rng):
    grad_fn = jax.value_and_grad(batch_loss_fn, has_aux=True)
    (_, batch_metrics), grads = grad_fn(train_state.params, train_state, batch_obs, loss_rng)
    grads = jax.lax.pmean(grads, axis_name="batch")
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, batch_metrics

  def step_train_fn(train_state, step_rng):
    epis_rng = random.split(step_rng, config["epis_per_step"])
    loss_rng = random.split(step_rng, config["n_batches"])
    step_obs = jax.vmap(episode_simul_fn, in_axes=(None,0))(train_state, jnp.stack(epis_rng))
    step_obs = step_obs.reshape(config["periods_per_step"], econ_model.obs_ss.shape[0])                # combine all periods in one axis
    step_obs = random.permutation(step_rng, step_obs, axis=0)                                          # reshuffle obs at random
    step_obs = step_obs.reshape(config["n_batches"], config["batch_size"] ,econ_model.obs_ss.shape[0]) # reshape to into batches
    train_state, step_metrics = jax.vmap(batch_train_fn, in_axes=(None,0,0), out_axes=(None,0), axis_name="batch")(train_state, step_obs, jnp.stack(loss_rng))
    mean_losses, max_losses, mean_accuracies, min_accuracies,_, _ = step_metrics
    mean_loss = jnp.mean(mean_losses)
    max_loss = jnp.mean(max_losses)
    mean_accuracy = jnp.mean(mean_accuracies)
    min_accuracy = jnp.min(min_accuracies)
    metrics = mean_loss, max_loss, mean_accuracy, min_accuracy
    return train_state, metrics

  def epoch_train_fn(train_state, epoch_rng):
    """Vectorise and repeat the update to complete an epoch, made aout of steps_per_epoch episodes."""
    epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
    train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
    return train_state, epoch_rng, epoch_metrics

  return epoch_train_fn

