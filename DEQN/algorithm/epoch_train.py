import jax
from jax import lax, random
from jax import numpy as jnp


def create_epoch_train_fn(econ_model, config):
    from DEQN.algorithm.loss import create_batch_loss_fn
    from DEQN.algorithm.simulation import create_episode_simul_fn

    episode_simul_fn = create_episode_simul_fn(econ_model, config)
    batch_loss_fn = create_batch_loss_fn(econ_model, config)

    def batch_train_fn(train_state, batch_obs, loss_rng):
        grad_fn = jax.value_and_grad(batch_loss_fn, has_aux=True)
        (_, batch_metrics), grads = grad_fn(train_state.params, train_state, batch_obs, loss_rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, batch_metrics

    def step_train_fn(train_state, step_rng):
        epis_rng = random.split(step_rng, config["epis_per_step"])
        loss_rng = random.split(step_rng, config["n_batches"])
        step_obs = jax.vmap(episode_simul_fn, in_axes=(None, 0))(train_state, jnp.stack(epis_rng))
        step_obs = step_obs.reshape(
            config["periods_per_step"], econ_model.state_ss.shape[0]
        )  # combine all periods in one axis
        step_obs = random.permutation(step_rng, step_obs, axis=0)  # reshuffle obs at random
        step_obs = step_obs.reshape(
            config["n_batches"], config["batch_size"], econ_model.state_ss.shape[0]
        )  # reshape to into batches
        train_state, step_metrics = jax.vmap(
            batch_train_fn, in_axes=(None, 0, 0), out_axes=(None, 0), axis_name="batch"
        )(train_state, step_obs, jnp.stack(loss_rng))
        mean_losses, mean_accuracies, min_accuracies, _, _ = step_metrics
        mean_loss = jnp.mean(mean_losses)
        mean_accuracy = jnp.mean(mean_accuracies)
        min_accuracy = jnp.min(min_accuracies)
        metrics = mean_loss, mean_accuracy, min_accuracy
        return train_state, metrics

    def epoch_train_fn(train_state, epoch_rng):
        """Vectorise and repeat the update to complete an epoch, made aout of steps_per_epoch episodes."""
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn


def create_fast_epoch_train_fn(econ_model, config):

    def step_train_fn(train_state, step_rng):
        apply_fn = train_state.apply_fn

        def step_loss(params, step_key):
            mc_key, traj_key = random.split(step_key)
            mc_shocks = econ_model.mc_shocks(mc_key, config["mc_draws"])  # (mc_draws, n_sectors)

            epis_keys = random.split(traj_key, config["epis_per_step"])  # keys per episode

            def run_episode(epis_key):
                init_obs = econ_model.initial_state(epis_key, config["init_range"])  # (obs_dim,)
                period_keys = random.split(epis_key, config["periods_per_epis"])  # per-period RNGs

                def run_period(obs, period_key):
                    policy = apply_fn(params, obs)

                    # Expectation with centralized MC shocks; block grads through dyn/expectation path
                    mc_nextobs = jax.vmap(
                        lambda shock: econ_model.step(
                            jax.lax.stop_gradient(obs),
                            jax.lax.stop_gradient(policy),
                            shock,
                        )
                    )(mc_shocks)
                    mc_nextpols = jax.vmap(lambda s: apply_fn(params, s))(mc_nextobs)
                    expect = jax.lax.stop_gradient(
                        jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
                    )

                    mean_loss, mean_accuracy, min_accuracy, _, _ = econ_model.loss(obs, expect, policy)

                    # Environment transition for next carry (no gradient through dynamics)
                    shock_sim = config["simul_vol_scale"] * econ_model.sample_shock(period_key)
                    obs_next = econ_model.step(
                        jax.lax.stop_gradient(obs),
                        jax.lax.stop_gradient(policy),
                        shock_sim,
                    )
                    return obs_next, (mean_loss, mean_accuracy, min_accuracy)

                _, (losses, mean_accs, min_accs) = lax.scan(run_period, init_obs, jnp.stack(period_keys))
                return jnp.mean(losses), jnp.mean(mean_accs), jnp.min(min_accs)

            losses, mean_accs, min_accs = jax.vmap(run_episode)(epis_keys)
            mean_loss = jnp.mean(losses)
            mean_accuracy = jnp.mean(mean_accs)
            min_accuracy = jnp.min(min_accs)
            metrics = mean_loss, mean_accuracy, min_accuracy
            return mean_loss, metrics

        grad_fn = jax.value_and_grad(step_loss, has_aux=True)
        (_, metrics), grads = grad_fn(train_state.params, step_rng)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    step_train_fn = jax.jit(step_train_fn, donate_argnums=(0,))

    def epoch_train_fn(train_state, epoch_rng):
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn
