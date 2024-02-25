from jax import numpy as jnp, lax, random
import jax

def create_batch_loss_fn(env, config):

  if config["proxy_mcsampler"] and config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) # get the policies for the entire obs batch.
      # batch_policies = jax.vmap(env.policy_loglinear)(batch_obs) # get the policies for the entire obs batch.
      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = env.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(env.step_loglinear, in_axes = (None,0))(obs, mc_shocks)
        # print("shape of mc_nextobs")
        mc_nextpols = jax.vmap(env.policy_loglinear)(mc_nextobs)
        # print("shape of mc_nexpols", )
        expect = jnp.mean(jax.vmap(env.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = env.loss(obs, expect, policy) # calculate loss
        return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           # average accross periods
      min_accuracy = jnp.min(min_accuracies)              # min accross periods and across eqs within period
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc # pass as auxiliary info
      # metrics = jnp.array([mean_losses, mean_accuracies, min_accuracies]) # pass as auxiliary info
      return mean_loss, metrics

  elif config["proxy_mcsampler"] and not config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) # get the policies for the entire obs batch.

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = env.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(env.step_loglinear, in_axes = (None,0))(obs, mc_shocks)
        mc_nextpols = train_state.apply_fn(params, mc_nextobs)
        expect = jnp.mean(jax.vmap(env.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = env.loss(obs, expect, policy) # calculate loss
        return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           # average accross periods
      min_accuracy = jnp.min(min_accuracies)              # min accross periods and across eqs within period
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc # pass as auxiliary info
      return mean_loss, metrics

  elif not config["proxy_mcsampler"] and config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) # get the policies for the entire obs batch.

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = env.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(env.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
        mc_nextpols = jax.vmap(env.policy_loglinear)(mc_nextobs)
        expect = jnp.mean(jax.vmap(env.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = env.loss(obs, expect, policy) # calculate loss
        return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

     # parallelize callculation of period_loss for the entire batch
      mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           # average accross periods
      min_accuracy = jnp.min(min_accuracies)              # min accross periods and across eqs within period
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc # pass as auxiliary info
      return mean_loss, metrics
  else:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) # get the policies for the entire obs batch.

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = env.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(env.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
        mc_nextpols = train_state.apply_fn(params, mc_nextobs)
        expect = jnp.mean(jax.vmap(env.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = env.loss(obs, expect, policy) # calculate loss
        return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           # average accross periods
      min_accuracy = jnp.min(min_accuracies)              # min accross periods and across eqs within period
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc # pass as auxiliary info
      return mean_loss, metrics

  return batch_loss_fn