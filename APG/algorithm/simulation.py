"""Episode simulation for the core APG algorithm."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random, vmap


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    obs: jax.Array
    info: jax.Array


class Metrics(NamedTuple):
    mean_loss: jax.Array
    mean_actor_loss: jax.Array
    mean_value_loss: jax.Array


def create_episode_simul_fn(env, config, use_terminal_value=None):
    """Create a function that simulates a single episode.

    Args:
        env: Differentiable environment with an economic transition model
        config: Must include periods_per_epis; rollout settings are optional
        use_terminal_value: Whether to bootstrap the return from a value head

    Returns:
        Function that takes (params, train_state, epis_rng) and returns
        (returns, trajectory, last_val)
    """
    periods_per_epis = config["periods_per_epis"]
    init_range = config.get("init_range", 5)
    init_range_a = config.get("init_range_a")
    simul_vol_scale = config.get("simul_vol_scale", 1.0)
    antithetic = bool(config.get("antithetic_episodes", False))
    if use_terminal_value is None:
        use_terminal_value = config.get("use_terminal_value", False)
    use_model_terminal_value = bool(config.get("use_model_terminal_value", False))
    use_lq_terminal_value = bool(config.get("use_lq_terminal_value", False))
    terminal_value_horizon = int(config.get("terminal_value_horizon", 512))
    rematerialize = bool(config.get("rematerialize_rollout", False)) or periods_per_epis >= 1024
    n_tails = sum(
        [bool(use_terminal_value), use_model_terminal_value, use_lq_terminal_value]
    )
    if n_tails > 1:
        raise ValueError(
            "learned critic, model terminal continuation, and LQ terminal value "
            "cannot be combined"
        )
    if terminal_value_horizon < 1:
        raise ValueError("terminal_value_horizon must be positive")

    def rollout(params, train_state, init_obs, shocks):
        runner_state = params, init_obs, 0, 1

        def period_step(runner_state, shock):
            params, obs, returns, discount = runner_state
            if use_terminal_value:
                action, value_notnorm = train_state.apply_fn(
                    params,
                    obs,
                    stop_critic_input_gradient=True,
                )
            else:
                action = train_state.apply_fn(params, obs)
                value_notnorm = jnp.zeros(action.shape[:-1], dtype=action.dtype)
            value = value_notnorm * env.value_ss
            reward = env.training_reward(obs, action)
            obs_next = env.transition(obs, action, shock)
            done = jnp.array(False)
            info = jnp.array([0.0])
            transition = Transition(done, action, value, reward, obs_next, info)
            returns = returns + discount * reward
            discount = env.discount_rate * discount
            runner_state = (params, obs_next, returns, discount)
            return runner_state, transition

        step_fn = jax.checkpoint(period_step) if rematerialize else period_step
        runner_state, trajectory = jax.lax.scan(step_fn, runner_state, shocks)

        _, last_obs, returns, discount = runner_state
        if use_lq_terminal_value:
            costate = jnp.asarray(config["lq_terminal_costate"], dtype=last_obs.dtype)
            P = jnp.asarray(config["lq_terminal_P"], dtype=last_obs.dtype)
            last_val = jnp.dot(costate, last_obs) + 0.5 * jnp.dot(last_obs, P @ last_obs)
            returns = returns + discount * last_val
        elif use_model_terminal_value:
            last_val = env.terminal_value(last_obs, terminal_value_horizon)
            returns = returns + discount * last_val
        elif use_terminal_value:
            _, last_val_notnorm = train_state.apply_fn(train_state.params, last_obs)
            last_val = last_val_notnorm * env.value_ss
            returns = returns + discount * last_val
        else:
            last_val = jnp.zeros_like(returns)

        return returns, trajectory, last_val

    def simul_episode(params, train_state, epis_rng):
        mode = config.get("initial_state_mode")
        if init_range_a is None:
            init_obs = env.initial_state(epis_rng, init_range, mode=mode)
        else:
            init_obs = env.initial_state(epis_rng, init_range, init_range_a, mode=mode)
        shocks = simul_vol_scale * vmap(env.sample_shock)(
            random.split(epis_rng, periods_per_epis)
        )
        returns, trajectory, last_val = rollout(params, train_state, init_obs, shocks)
        if not antithetic:
            return returns, trajectory, last_val
        returns_minus, _, last_val_minus = rollout(
            params, train_state, init_obs, -shocks
        )
        return 0.5 * (returns + returns_minus), trajectory, 0.5 * (last_val + last_val_minus)

    return simul_episode
