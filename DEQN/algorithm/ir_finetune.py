import jax
from jax import lax, random
from jax import numpy as jnp


def _validate_ir_finetune_config(config):
    min_shock = float(config["ir_finetune_min_shock_size"])
    max_shock = float(config["ir_finetune_max_shock_size"])

    if min_shock < 0:
        raise ValueError("ir_finetune_min_shock_size must be non-negative.")
    if max_shock <= min_shock:
        raise ValueError("ir_finetune_max_shock_size must be greater than ir_finetune_min_shock_size.")
    if max_shock >= 100:
        raise ValueError("ir_finetune_max_shock_size must be below 100 percent.")

    states_to_shock = list(config.get("states_to_shock", []))
    if not states_to_shock:
        raise ValueError("IR fine-tuning requires config['states_to_shock'] to contain at least one state index.")

    if int(config["periods_per_epis"]) < 1:
        raise ValueError("IR fine-tuning requires periods_per_epis >= 1.")

    periods_per_step = config["periods_per_step"]
    batch_size = config["batch_size"]
    if periods_per_step % batch_size != 0:
        raise ValueError(
            "IR fine-tuning periods_per_step must be divisible by batch_size. "
            f"Got periods_per_step={periods_per_step}, batch_size={batch_size}."
        )


def _apply_gir_style_shock(
    econ_model,
    state_normalized,
    state_idx,
    shock_size_pct,
    shock_sign,
    symmetric_shocks,
):
    shock_size = shock_size_pct / 100.0
    state_notnorm = state_normalized * econ_model.state_sd + econ_model.state_ss

    if shock_sign == "neg":
        log_shock = jnp.log(1 - shock_size)
    elif symmetric_shocks:
        log_shock = -jnp.log(1 - shock_size)
    else:
        log_shock = jnp.log(1 + shock_size)

    shocked_state_notnorm = state_notnorm.at[state_idx].add(log_shock)
    return (shocked_state_notnorm - econ_model.state_ss) / econ_model.state_sd


def create_ir_finetune_epoch_train_fn(econ_model, config):
    from DEQN.algorithm.loss import create_batch_loss_fn
    from DEQN.algorithm.simulation import create_episode_simul_fn

    _validate_ir_finetune_config(config)

    base_episode_simul_fn = create_episode_simul_fn(econ_model, config)
    batch_loss_fn = create_batch_loss_fn(econ_model, config)

    states_to_shock = jnp.array(config["states_to_shock"])
    min_shock = float(config["ir_finetune_min_shock_size"])
    max_shock = float(config["ir_finetune_max_shock_size"])
    symmetric_shocks = bool(config.get("gir_symmetric_shocks", True))
    shock_dimension = jnp.atleast_1d(econ_model.sample_shock(random.PRNGKey(0))).shape[-1]
    zero_shock = jnp.zeros((shock_dimension,), dtype=econ_model.state_ss.dtype)
    rollout_length = int(config["periods_per_epis"])

    def batch_train_fn(train_state, batch_obs, loss_rng):
        grad_fn = jax.value_and_grad(batch_loss_fn, has_aux=True)
        (_, batch_metrics), grads = grad_fn(train_state.params, train_state, batch_obs, loss_rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, batch_metrics

    def zero_shock_rollout(train_state, initial_obs):
        def period_step(obs, _unused):
            policy = train_state.apply_fn(train_state.params, obs)
            obs_next = econ_model.step(obs, policy, zero_shock)
            return obs_next, obs_next

        if rollout_length == 1:
            return initial_obs[None, :]

        _, future_obs = lax.scan(period_step, initial_obs, jnp.arange(rollout_length - 1))
        return jnp.concatenate([initial_obs[None, :], future_obs], axis=0)

    def shocked_rollouts_from_base_episode(train_state, epis_rng):
        base_rng, draw_rng, state_rng, shock_rng = random.split(epis_rng, 4)
        base_obs = base_episode_simul_fn(train_state, base_rng)
        base_idx = random.randint(draw_rng, shape=(), minval=0, maxval=base_obs.shape[0])
        base_state = base_obs[base_idx]

        state_choice_idx = random.randint(state_rng, shape=(), minval=0, maxval=states_to_shock.shape[0])
        state_idx = states_to_shock[state_choice_idx]
        shock_size_pct = random.uniform(shock_rng, shape=(), minval=min_shock, maxval=max_shock)

        neg_initial_obs = _apply_gir_style_shock(
            econ_model,
            base_state,
            state_idx,
            shock_size_pct,
            "neg",
            symmetric_shocks,
        )
        pos_initial_obs = _apply_gir_style_shock(
            econ_model,
            base_state,
            state_idx,
            shock_size_pct,
            "pos",
            symmetric_shocks,
        )

        neg_rollout = zero_shock_rollout(train_state, neg_initial_obs)
        pos_rollout = zero_shock_rollout(train_state, pos_initial_obs)
        return jnp.concatenate([neg_rollout, pos_rollout], axis=0)

    def step_train_fn(train_state, step_rng):
        epis_rng = random.split(step_rng, config["epis_per_step"])
        loss_rng = random.split(step_rng, config["n_batches"])
        step_obs = jax.vmap(shocked_rollouts_from_base_episode, in_axes=(None, 0))(
            train_state,
            jnp.stack(epis_rng),
        )
        step_obs = step_obs.reshape(config["periods_per_step"], econ_model.state_ss.shape[0])
        step_obs = random.permutation(step_rng, step_obs, axis=0)
        step_obs = step_obs.reshape(config["n_batches"], config["batch_size"], econ_model.state_ss.shape[0])
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
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn
