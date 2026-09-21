"""Euler-error diagnostic for the shared RBC model.

Uses `Model.euler_residual` / `Model.euler_error`. Occupancy and MC draws match
the DEQN eval config so APG and DEQN report the same residual.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax, random


def create_euler_eval_fn(
    model,
    policy_fn: Callable,
    periods_per_epis: int,
    n_epis: int,
    mc_draws: int,
    init_range: int = 5,
    simul_vol_scale: float = 1.0,
    init_range_a=None,
):
    def simulate_episode(params, epis_rng):
        obs = model.initial_state(
            epis_rng, init_range, init_range_a=init_range_a, mode="box"
        )
        period_rngs = random.split(epis_rng, periods_per_epis)

        def period_step(obs, period_rng):
            policy = policy_fn(params, obs)
            shock = simul_vol_scale * model.sample_shock(period_rng)
            return model.step(obs, policy, shock), obs

        _, states = lax.scan(period_step, obs, period_rngs)
        return states

    def period_euler(params, state, period_rng):
        policy = policy_fn(params, state)
        expect = model.monte_carlo_expect(
            state,
            policy,
            lambda obs: policy_fn(params, obs),
            period_rng,
            mc_draws,
        )
        mean_loss, mean_accuracy, min_accuracy, _, _ = model.euler_error(state, policy, expect)
        return mean_loss, mean_accuracy, min_accuracy

    def episode_euler(params, key):
        sim_rng, loss_rng = random.split(key)
        states = simulate_episode(params, sim_rng)
        period_rngs = random.split(loss_rng, states.shape[0])
        mean_loss, mean_accuracy, min_accuracy = jax.vmap(period_euler, in_axes=(None, 0, 0))(
            params, states, period_rngs
        )
        return jnp.mean(mean_loss), jnp.mean(mean_accuracy), jnp.min(min_accuracy)

    @jax.jit
    def euler_eval_fn(params, rng):
        keys = random.split(rng, n_epis)
        mean_loss, mean_accuracy, min_accuracy = jax.vmap(episode_euler, in_axes=(None, 0))(params, keys)
        return jnp.mean(mean_loss), jnp.mean(mean_accuracy), jnp.min(min_accuracy)

    return euler_eval_fn


def euler_metrics_to_dict(euler_metrics):
    mean_loss, mean_accuracy, min_accuracy = euler_metrics
    return {
        "euler_loss": float(mean_loss),
        "euler_acc": float(mean_accuracy),
        "euler_min_acc": float(min_accuracy),
    }


def print_euler_metrics(euler_metrics, prefix="  Euler"):
    mean_loss, mean_accuracy, min_accuracy = euler_metrics
    print(
        f"{prefix}: loss={float(mean_loss):.6f} "
        f"acc={float(mean_accuracy):.4f} "
        f"min={float(min_accuracy):.4f}",
        flush=True,
    )


def create_target_shift_eval_fn(
    model,
    policy_fn: Callable,
    periods_per_epis: int,
    n_epis: int,
    mc_draws: int,
    init_range: int = 5,
    simul_vol_scale: float = 1.0,
    init_range_a=None,
):
    """Mean |Euler residual| at eval states vs at their inner next-state draws."""

    def simulate_episode(params, epis_rng):
        obs = model.initial_state(
            epis_rng, init_range, init_range_a=init_range_a, mode="box"
        )
        period_rngs = random.split(epis_rng, periods_per_epis)

        def period_step(obs, period_rng):
            policy = policy_fn(params, obs)
            shock = simul_vol_scale * model.sample_shock(period_rng)
            return model.step(obs, policy, shock), obs

        _, states = lax.scan(period_step, obs, period_rngs)
        return states

    def abs_residual(params, state, rng):
        policy = policy_fn(params, state)
        expect = model.monte_carlo_expect(
            state,
            policy,
            lambda obs: policy_fn(params, obs),
            rng,
            mc_draws,
        )
        residual = jnp.reshape(model.euler_residual(state, policy, expect), (-1,))
        return jnp.abs(residual)

    def period_shift(params, state, period_rng):
        state_rng, next_rng, inner_rng = random.split(period_rng, 3)
        abs_at_state = abs_residual(params, state, state_rng)
        policy = policy_fn(params, state)
        shocks = model.mc_shocks(next_rng, mc_draws)
        next_states = jax.vmap(lambda shock: model.step(state, policy, shock))(shocks)
        next_rngs = random.split(inner_rng, mc_draws)
        abs_at_next = jax.vmap(abs_residual, in_axes=(None, 0, 0))(params, next_states, next_rngs)
        return jnp.mean(abs_at_state), jnp.max(abs_at_state), jnp.mean(abs_at_next)

    def episode_shift(params, key):
        sim_rng, loss_rng = random.split(key)
        states = simulate_episode(params, sim_rng)
        period_rngs = random.split(loss_rng, states.shape[0])
        mean_abs, max_abs, next_abs = jax.vmap(period_shift, in_axes=(None, 0, 0))(
            params, states, period_rngs
        )
        return jnp.mean(mean_abs), jnp.max(max_abs), jnp.mean(next_abs)

    @jax.jit
    def target_shift_eval_fn(params, rng):
        keys = random.split(rng, n_epis)

        def episode_body(_, key):
            return None, episode_shift(params, key)

        _, (mean_abs, max_abs, next_abs) = lax.scan(episode_body, None, keys)
        mean_at_states = jnp.mean(mean_abs)
        max_at_states = jnp.max(max_abs)
        mean_at_next = jnp.mean(next_abs)
        ratio = mean_at_next / jnp.maximum(mean_at_states, jnp.asarray(1e-12, dtype=mean_at_states.dtype))
        return mean_at_states, max_at_states, mean_at_next, ratio

    return target_shift_eval_fn


def target_shift_metrics_to_dict(metrics):
    mean_abs, max_abs, next_abs, ratio = metrics
    return {
        "mean_abs_residual": float(mean_abs),
        "max_abs_residual": float(max_abs),
        "mean_abs_residual_next": float(next_abs),
        "ratio": float(ratio),
    }
