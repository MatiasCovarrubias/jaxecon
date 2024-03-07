from jax import numpy as jnp, lax, random
import jax


def create_batch_loss_fn_simple(econ_model, config):

  def batch_loss_fn(params, train_state, batch_obs, loss_rng):
    """Loss function of a batch of obs."""
    period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
    batch_policies = train_state.apply_fn(params, batch_obs) 

    def period_loss(obs, policy, period_mc_rng):
      """Loss function for an individual period."""
      mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
      mc_nextobs = jax.vmap(econ_model.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
      mc_nextpols = train_state.apply_fn(params, mc_nextobs)
      expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
      mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
      return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

    # parallelize callculation of period_loss for the entire batch
    losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
    mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
    mean_loss = jnp.mean(mean_losses)                   # average accross periods
    max_loss = jnp.max(max_losses)                      # max accross periods
    mean_accuracy = jnp.mean(mean_accuracies)           
    min_accuracy = jnp.min(min_accuracies)              
    mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
    min_accs_foc = jnp.min(min_accs_foc,axis=0) 
    loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
    return mean_loss, loss_metrics

  return batch_loss_fn


def create_batch_loss_fn_precomputed_expects(econ_model):

  def batch_loss_fn(params, train_state, batch_obs):
    """Loss function for a batch of observations."""
    batch_policies, batch_expects = train_state.apply_fn(params, batch_obs, freeze_expects=True) 

    # parallelize callculation of period_loss for the entire batch
    losses_metrics = jax.vmap(econ_model.loss)(batch_obs, batch_expects, batch_policies)
    mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
    mean_loss = jnp.mean(mean_losses)                   # average accross periods
    max_loss = jnp.max(max_losses)                      # max accross periods
    mean_accuracy = jnp.mean(mean_accuracies)           
    min_accuracy = jnp.min(min_accuracies)              
    mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
    min_accs_foc = jnp.min(min_accs_foc,axis=0) 
    loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
    return mean_loss, loss_metrics

  return batch_loss_fn

def create_batch_loss_fn_proxied(econ_model, config):

  if config["proxy_loss"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 
  
      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step_loglinear, in_axes = (None,0))(obs, mc_shocks)
        mc_nextpols = jax.vmap(econ_model.policy_loglinear)(mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics

  else:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
        mc_nextpols = train_state.apply_fn(params, mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics

  return batch_loss_fn

def create_batch_loss_fn_flexibleproxy(econ_model, config):

  if config["proxy_mcsampler"] and config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step_loglinear, in_axes = (None,0))(obs, mc_shocks)
        mc_nextpols = jax.vmap(econ_model.policy_loglinear)(mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics

  elif config["proxy_mcsampler"] and not config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step_loglinear, in_axes = (None,0))(obs, mc_shocks)
        mc_nextpols = train_state.apply_fn(params, mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics

  elif not config["proxy_mcsampler"] and config["proxy_futurepol"]:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
        mc_nextpols = jax.vmap(econ_model.policy_loglinear)(mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics
  else:
    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
      """Loss function of a batch of obs."""
      period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
      batch_policies = train_state.apply_fn(params, batch_obs) 

      def period_loss(obs, policy, period_mc_rng):
        """Loss function for an individual period."""
        mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
        mc_nextobs = jax.vmap(econ_model.step, in_axes = (None,None,0))(obs, policy, mc_shocks)
        mc_nextpols = train_state.apply_fn(params, mc_nextobs)
        expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
        mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy) 
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

      # parallelize callculation of period_loss for the entire batch
      losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
      mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
      mean_loss = jnp.mean(mean_losses)                   # average accross periods
      max_loss = jnp.mean(max_losses)                     # max accross periods
      mean_accuracy = jnp.mean(mean_accuracies)           
      min_accuracy = jnp.min(min_accuracies)              
      mean_accs_foc = jnp.mean(mean_accs_foc,axis=0)      # average across period for each set of focs
      min_accs_foc = jnp.min(min_accs_foc,axis=0)
      metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc 
      return mean_loss, metrics

  return batch_loss_fn

