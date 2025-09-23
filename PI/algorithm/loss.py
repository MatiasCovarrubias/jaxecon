import jax
from jax import numpy as jnp
from jax import random


def create_batch_loss_fn_policies(econ_model):

    def batch_loss_fn(params, train_state, batch_obs):
        """Loss function for a batch of observations."""
        batch_policies, batch_expects = train_state.apply_fn(params, batch_obs, freeze_expects=True)

        # parallelize callculation of period_loss for the entire batch
        losses_metrics = jax.vmap(econ_model.loss)(batch_obs, batch_expects, batch_policies)
        mean_losses, max_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        max_loss = jnp.max(max_losses)  # max accross periods
        mean_accuracy = jnp.mean(mean_accuracies)
        min_accuracy = jnp.min(min_accuracies)
        mean_accs_foc = jnp.mean(mean_accs_foc, axis=0)  # average across period for each set of focs
        min_accs_foc = jnp.min(min_accs_foc, axis=0)
        loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc
        return mean_loss, loss_metrics

    return batch_loss_fn


def create_batch_loss_fn_expects(econ_model, config):

    def batch_loss_fn(params, train_state, batch_obs, loss_rng):

        period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])

        """Loss function of a batch of obs."""
        batch_policies, batch_expects = train_state.apply_fn(
            params, batch_obs, freeze_policies=True
        )  # get the policies for the entire obs batch.

        def period_loss(obs, policy, expect, period_mc_rng):
            """Loss function for an individual period."""
            mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
            mc_nextobs = jax.vmap(econ_model.step, in_axes=(None, None, 0))(obs, policy, mc_shocks)
            mc_nextpols = train_state.apply_fn(params, mc_nextobs)
            true_expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
            mean_loss = jnp.mean((1 - expect / true_expect) ** 2)
            max_loss = jnp.max((1 - expect / true_expect) ** 2)
            mean_accuracy = jnp.mean(1 - jnp.abs(1 - expect / true_expect))
            min_accuracy = jnp.min(1 - jnp.abs(1 - expect / true_expect))
            return mean_loss, max_loss, mean_accuracy, min_accuracy

        # parallelize callculation of period_loss for the entire batch
        mean_losses, max_losses, mean_accuracies, min_accuracies = jax.vmap(period_loss)(
            batch_obs, batch_policies, batch_expects, jnp.stack(period_mc_rngs)
        )
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        max_loss = jnp.max(max_losses)  # max accross periods
        mean_accuracy = jnp.mean(mean_accuracies)  # average accross periods
        min_accuracy = jnp.min(min_accuracies)  # min accross periods and across eqs within period
        loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy
        return mean_loss, loss_metrics

    return batch_loss_fn


def create_batch_loss_fn_pretrain_policies(econ_model):

    def batch_loss_fn(params, train_state, batch_obs):
        """Loss function of a batch of obs."""
        batch_policies, _ = train_state.apply_fn(
            params, batch_obs, freeze_expects=True
        )  # get the policies for the entire obs batch.
        batch_policies_parent = jax.vmap(econ_model.policy_loglinear)(batch_obs)

        def period_loss(policy, policy_parent):
            """Loss function for an individual period."""
            mean_loss = jnp.mean((1 - policy / policy_parent) ** 2)
            max_loss = jnp.max((1 - policy / policy_parent) ** 2)
            mean_accuracy = jnp.mean(1 - jnp.abs(1 - policy / policy_parent))
            min_accuracy = jnp.min(1 - jnp.abs(1 - policy / policy_parent))
            return mean_loss, max_loss, mean_accuracy, min_accuracy

        # parallelize callculation of period_loss for the entire batch
        mean_losses, max_losses, mean_accuracies, min_accuracies = jax.vmap(period_loss)(
            batch_policies, batch_policies_parent
        )
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        max_loss = jnp.max(max_losses)  # max accross periods
        mean_accuracy = jnp.mean(mean_accuracies)  # average accross periods
        min_accuracy = jnp.min(min_accuracies)  # min accross periods and across eqs within period
        loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy
        return mean_loss, loss_metrics

    return batch_loss_fn


def create_batch_loss_fn_pretrain_expects(econ_model, config):

    def batch_loss_fn(params, train_state, batch_obs, loss_rng):

        period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])

        """Loss function of a batch of obs."""
        _, batch_expects = train_state.apply_fn(
            params, batch_obs, freeze_policies=True
        )  # get the policies for the entire obs batch.

        def period_loss(obs, expect, period_mc_rng):
            """Loss function for an individual period."""
            mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
            mc_nextobs = jax.vmap(econ_model.step_loglinear, in_axes=(None, 0))(obs, mc_shocks)
            mc_nextpols = jax.vmap(econ_model.policy_loglinear)(mc_nextobs)
            true_expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
            mean_loss = jnp.mean((1 - expect / true_expect) ** 2)
            max_loss = jnp.max((1 - expect / true_expect) ** 2)
            mean_accuracy = jnp.mean(1 - jnp.abs(1 - expect / true_expect))
            min_accuracy = jnp.min(1 - jnp.abs(1 - expect / true_expect))
            return mean_loss, max_loss, mean_accuracy, min_accuracy

        # parallelize callculation of period_loss for the entire batch
        mean_losses, max_losses, mean_accuracies, min_accuracies = jax.vmap(period_loss)(
            batch_obs, batch_expects, jnp.stack(period_mc_rngs)
        )
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        max_loss = jnp.max(max_losses)  # max accross periods
        mean_accuracy = jnp.mean(mean_accuracies)  # average accross periods
        min_accuracy = jnp.min(min_accuracies)  # min accross periods and across eqs within period
        loss_metrics = mean_loss, max_loss, mean_accuracy, min_accuracy
        return mean_loss, loss_metrics

    return batch_loss_fn
