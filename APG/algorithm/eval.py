from jax import numpy as jnp, random
import jax
from .simulation import create_simul_episode_fn


def create_episode_loss_fn_eval(env, config, eval_periods_per_epis):
    simul_episode = create_simul_episode_fn(env, eval_periods_per_epis)

    # Define function that gives targets for value updates
    def get_targets(trajectory, last_val):

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

    return episode_loss_fn


def get_eval_fn(env, config):
    episode_loss_fn = create_episode_loss_fn_eval(env, config, config["eval_periods_per_epis"])

    def episode_grads_and_metrics(train_state, epis_rng):
        grad_fn = jax.value_and_grad(episode_loss_fn, has_aux=True)
        loss_metrics, grads = grad_fn(train_state.params, train_state, epis_rng)
        grads = jax.lax.pmean(grads, axis_name="episodes")
        grad_mean = jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree_map(jnp.mean, grads))))
        grad_max = jnp.max(jnp.array(jax.tree_util.tree_leaves(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))))
        grad_metrics = (grad_mean, grad_max)
        episode_metrics = (loss_metrics, grad_metrics)
        return episode_metrics

    def eval_fn(train_state, eval_rng):
        epis_rng = random.split(eval_rng, config["eval_n_epis"])
        loss_metrics, grad_metrics = jax.vmap(
            episode_grads_and_metrics, in_axes=(None, 0), out_axes=(0), axis_name="episodes"
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
