import jax
from jax import lax, random
from jax import numpy as jnp


def _maybe_stop_gradient(x, enabled):
    return jax.lax.stop_gradient(x) if enabled else x


def create_epoch_train_fn(econ_model, config):
    update = config.get("deqn_update", "residual")
    if update in ("plain_euler", "plain_envelope"):
        return create_plain_deqn_epoch_train_fn(econ_model, config)
    if update != "residual":
        raise ValueError(
            "deqn_update must be 'residual', 'plain_euler', or "
            "'plain_envelope'"
        )

    from DEQN.algorithm.loss import create_batch_loss_fn, create_batch_loss_fn_nostopgrad
    from DEQN.algorithm.simulation import create_step_state_sampler

    step_state_sampler = create_step_state_sampler(econ_model, config)
    loss_factory = create_batch_loss_fn if config.get("deqn_stop_gradient", True) else create_batch_loss_fn_nostopgrad
    batch_loss_fn = loss_factory(econ_model, config)

    def batch_train_fn(train_state, batch_obs, loss_rng):
        grad_fn = jax.value_and_grad(batch_loss_fn, has_aux=True)
        (_, batch_metrics), grads = grad_fn(train_state.params, train_state, batch_obs, loss_rng)
        grads = jax.lax.pmean(grads, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, batch_metrics

    def step_train_fn(train_state, step_rng):
        loss_rng = random.split(step_rng, config["n_batches"])
        step_obs = step_state_sampler(train_state, step_rng)
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


def create_plain_deqn_epoch_train_fn(econ_model, config):
    """Train with the stopped, linear action-coordinate Euler moment.

    This opt-in Stone--Geary comparison operator reuses the exact current
    states and realized next shocks from each simulated trajectory. It applies
    discounted period weights and averages over independent trajectories in
    the same way APG averages finite-horizon returns.
    """
    from DEQN.algorithm.loss import plain_deqn_euler_surrogate
    from DEQN.algorithm.simulation import sample_episode_shocks
    bridge_depth = config.get("envelope_depth")
    if config.get("deqn_update") == "plain_envelope":
        if bridge_depth is None:
            raise ValueError(
                "plain_envelope requires a positive envelope_depth"
            )
        bridge_depth = int(bridge_depth)
        if bridge_depth < 1:
            raise ValueError(
                "plain_envelope requires a positive envelope_depth"
            )

    if int(econ_model.n_sectors) != 1 or float(econ_model.i_min_frac) != 0:
        raise ValueError(
            "plain_euler is implemented only for the smooth one-sector RBC"
        )
    if config.get("state_sampling", {}).get("mode", "occupancy") != "occupancy":
        raise ValueError("plain_euler requires occupancy state sampling")

    periods = int(config["periods_per_epis"])
    episodes = int(config["epis_per_step"])
    antithetic = bool(config.get("antithetic_episodes", False))
    init_range = config.get("init_range", 0)
    init_range_a = config.get("init_range_a")
    init_mode = config.get("initial_state_mode")
    vol_scale = config.get("simul_vol_scale", 1.0)
    discounts = econ_model.discount_rate ** jnp.arange(
        periods, dtype=econ_model.precision
    )
    use_lq_terminal = bool(config.get("use_lq_terminal_value", False))
    terminal_costate = None
    if use_lq_terminal:
        lq_costate = jnp.asarray(
            config["lq_terminal_costate"], dtype=econ_model.precision
        )
        lq_P = jnp.asarray(
            config["lq_terminal_P"], dtype=econ_model.precision
        )

        def terminal_costate(next_states):
            normalized_gradient = lq_costate + next_states @ lq_P.T
            capital, _ = econ_model._capital_and_productivity(next_states)
            return (
                normalized_gradient[..., : econ_model.n_sectors]
                / (
                    capital
                    * econ_model.state_sd[: econ_model.n_sectors]
                )
            )

    def initial_state(key):
        kwargs = {}
        if init_range_a is not None:
            kwargs["init_range_a"] = init_range_a
        if init_mode is not None:
            kwargs["mode"] = init_mode
        return econ_model.initial_state(key, init_range, **kwargs)

    def rollout(params, apply_fn, state, shocks):
        def period_step(current_state, shock):
            policy = apply_fn(params, current_state)
            next_state = econ_model.step(current_state, policy, shock)
            return next_state, current_state

        _, states = lax.scan(period_step, state, shocks)
        return states

    def step_loss(params, train_state, step_rng):
        episode_keys = random.split(step_rng, episodes)

        def one_path(key):
            state = initial_state(key)
            shocks = sample_episode_shocks(
                econ_model, key, periods, vol_scale
            )
            plus_states = rollout(
                params, train_state.apply_fn, state, shocks
            )
            if not antithetic:
                return plus_states[None, ...], shocks[None, ...]
            minus_states = rollout(
                params, train_state.apply_fn, state, -shocks
            )
            return (
                jnp.stack((plus_states, minus_states)),
                jnp.stack((shocks, -shocks)),
            )

        states, shocks = jax.vmap(one_path)(episode_keys)
        paths_per_episode = 2 if antithetic else 1
        path_count = episodes * paths_per_episode
        flat_states = states.reshape(
            path_count * periods, econ_model.dim_states
        )
        flat_shocks = shocks.reshape(
            path_count * periods, econ_model.n_sectors
        )
        weights = jnp.tile(discounts, path_count)
        terminal_mask = jnp.tile(
            jnp.arange(periods) == periods - 1, path_count
        )
        if bridge_depth is None:
            loss, aux = plain_deqn_euler_surrogate(
                params,
                train_state.apply_fn,
                econ_model,
                flat_states,
                flat_shocks,
                sample_weights=weights,
                normalizer=path_count,
                continuation_override_fn=terminal_costate,
                continuation_override_mask=(
                    terminal_mask if use_lq_terminal else None
                ),
            )
        else:
            from DEQN.algorithm.loss import plain_deqn_envelope_surrogate

            path_states = flat_states.reshape(
                path_count, periods, econ_model.dim_states
            )
            path_shocks = flat_shocks.reshape(
                path_count, periods, econ_model.n_sectors
            )
            loss, aux = plain_deqn_envelope_surrogate(
                params,
                train_state.apply_fn,
                econ_model,
                path_states,
                path_shocks,
                depth=bridge_depth,
                sample_weights=weights.reshape(path_count, periods),
                normalizer=path_count,
                continuation_override_fn=terminal_costate,
                continuation_override_mask=(
                    terminal_mask if use_lq_terminal else None
                ),
            )
        denominator = jnp.maximum(
            econ_model.beta * aux.expected_continuation,
            jnp.asarray(1e-12, dtype=econ_model.precision),
        )
        ratio = aux.L / denominator
        metrics = (
            loss,
            jnp.mean(1 - jnp.abs(ratio)),
            jnp.min(1 - jnp.abs(ratio)),
        )
        return loss, metrics

    def step_train_fn(train_state, step_rng):
        (_, metrics), grads = jax.value_and_grad(
            step_loss, has_aux=True
        )(train_state.params, train_state, step_rng)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def epoch_train_fn(train_state, epoch_rng):
        epoch_rng, *step_rngs = random.split(
            epoch_rng, config["steps_per_epoch"] + 1
        )
        train_state, epoch_metrics = lax.scan(
            step_train_fn, train_state, jnp.stack(step_rngs)
        )
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn


def create_optimal_fast_epoch_train_fn(econ_model, config):

    from DEQN.algorithm.simulation import create_episode_simul_fn

    episode_simul_fn = create_episode_simul_fn(econ_model, config)
    stop_grad = config.get("deqn_stop_gradient", True)

    def step_train_fn(train_state, step_rng):
        apply_fn = train_state.apply_fn

        simul_key, mc_key = random.split(step_rng)

        epis_keys = random.split(simul_key, config["epis_per_step"])  # keys per episode

        # Simulate all episodes for this step (outside the gradient path)
        step_obs = jax.vmap(episode_simul_fn, in_axes=(None, 0))(train_state, jnp.stack(epis_keys))

        # Flatten episodes × periods into a single batch of observations
        periods_per_step = config.get("periods_per_step", config["epis_per_step"] * config["periods_per_epis"])
        step_obs = step_obs.reshape(periods_per_step, econ_model.state_ss.shape[0])

        def step_loss(params):
            # Policies for all observations in one batched call
            policies = apply_fn(params, step_obs)

            # Shared MC shocks for expectation (centralized MC)
            mc_shocks = econ_model.mc_shocks(mc_key, config["mc_draws"])  # (mc_draws, shock_dim)

            # Compute next observations for all obs under each MC shock
            def step_with_shock(shock):
                return jax.vmap(
                    lambda obs, pol: econ_model.step(
                        _maybe_stop_gradient(obs, stop_grad),
                        _maybe_stop_gradient(pol, stop_grad),
                        shock,
                    )
                )(step_obs, policies)

            mc_nextobs = jax.vmap(step_with_shock)(mc_shocks)  # (mc_draws, batch, obs_dim)

            # Policies at next observations in a single batched call
            batch_size = mc_nextobs.shape[1]
            obs_dim = mc_nextobs.shape[2]
            mc_nextobs_2d = mc_nextobs.reshape(config["mc_draws"] * batch_size, obs_dim)
            mc_nextpols_2d = apply_fn(params, mc_nextobs_2d)
            mc_nextpols = mc_nextpols_2d.reshape(mc_nextobs.shape[0], mc_nextobs.shape[1], -1)

            # Expected objects (stop gradient through expectation/dynamics)
            expect_real = jax.vmap(jax.vmap(econ_model.expect_realization))(
                mc_nextobs, mc_nextpols
            )  # (mc_draws, batch, ...)
            expect = _maybe_stop_gradient(jnp.mean(expect_real, axis=0), stop_grad)  # (batch, ...)

            # Per-observation losses/accuracies; differentiate w.r.t. policy/params only
            mean_loss_i, mean_acc_i, min_acc_i, _, _ = jax.vmap(lambda o, e, p: econ_model.loss(o, e, p))(
                step_obs, expect, policies
            )

            mean_loss = jnp.mean(mean_loss_i)
            mean_accuracy = jnp.mean(mean_acc_i)
            min_accuracy = jnp.min(min_acc_i)
            metrics = mean_loss, mean_accuracy, min_accuracy
            return mean_loss, metrics

        grad_fn = jax.value_and_grad(step_loss, has_aux=True)
        (_, metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def epoch_train_fn(train_state, epoch_rng):
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train_fn, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train_fn


def create_fast_epoch_train_fn(econ_model, config):

    stop_grad = config.get("deqn_stop_gradient", True)

    def step_train_fn(train_state, step_rng):
        apply_fn = train_state.apply_fn

        def step_loss(params, step_key):
            mc_key, traj_key = random.split(step_key)
            mc_shocks = econ_model.mc_shocks(mc_key, config["mc_draws"])  # (mc_draws, n_sectors)

            epis_keys = random.split(traj_key, config["epis_per_step"])  # keys per episode

            def run_episode(epis_key):
                init_range_a = config.get("init_range_a")
                mode = config.get("initial_state_mode")
                kwargs = {}
                if init_range_a is not None:
                    kwargs["init_range_a"] = init_range_a
                if mode is not None:
                    kwargs["mode"] = mode
                init_obs = econ_model.initial_state(epis_key, config["init_range"], **kwargs)
                period_keys = random.split(epis_key, config["periods_per_epis"])  # per-period RNGs

                def run_period(obs, period_key):
                    policy = apply_fn(params, obs)

                    # Expectation with centralized MC shocks; block grads through dyn/expectation path
                    mc_nextobs = jax.vmap(
                        lambda shock: econ_model.step(
                            _maybe_stop_gradient(obs, stop_grad),
                            _maybe_stop_gradient(policy, stop_grad),
                            shock,
                        )
                    )(mc_shocks)
                    mc_nextpols = jax.vmap(lambda s: apply_fn(params, s))(mc_nextobs)
                    expect = _maybe_stop_gradient(
                        jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0),
                        stop_grad,
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
