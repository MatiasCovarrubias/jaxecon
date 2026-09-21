"""Shared simulation and metrics for a policy ``state -> control``.

A neural net and an interpolated time-iteration grid are the same kind of
object here. Every policy is simulated on one common sample of initial states
and shocks. Welfare is the mean discounted utility on that path. The residual
is the model's Euler residual at the states visited on that same path, with
one Monte Carlo shock bank shared by every policy.
"""

import jax
from jax import numpy as jnp
from jax import random


def steady_state_policy(model):
    """Saving rate fixed at the deterministic steady state."""

    def policy(state):
        del state
        return model.control_ss

    return policy


def draw_sample(model, episodes, periods, mc_draws, seed, rng=None):
    """Common initial states, shock paths, and Euler shock bank."""
    if rng is None:
        rng = random.PRNGKey(int(seed))
    state_key, shock_key, mc_key = random.split(rng, 3)
    states0 = jax.vmap(model.initial_state)(random.split(state_key, episodes))
    shocks = jax.vmap(lambda key: jax.vmap(model.sample_shock)(random.split(key, periods)))(
        random.split(shock_key, episodes)
    )
    mc_shocks = jax.vmap(model.sample_shock)(random.split(mc_key, mc_draws))
    return states0, shocks, mc_shocks


def path(model, policy, state, shocks):
    """One episode. Returns the final state, the decision states, and rewards.

    ``policy(state)`` is any map to a control. APG passes a live network, DEQN
    a frozen one, and evaluation either of those or the time-iteration grid.
    """

    def step(carry, shock):
        control = policy(carry)
        reward = model.utility(carry, control)
        return model.transition(carry, control, shock), (carry, reward)

    final, (states, rewards) = jax.lax.scan(step, state, shocks)
    return final, states, rewards


def simulate(model, policy, states0, shocks):
    """Decision states and period rewards. ``states`` lines up with ``shocks``."""

    def episode(state, episode_shocks):
        _, states, rewards = path(model, policy, state, episode_shocks)
        return states, rewards

    return jax.vmap(episode)(states0, shocks)


def welfare(model, rewards):
    """Mean over episodes of discounted utility."""
    discounts = model.discount_rate ** jnp.arange(rewards.shape[-1], dtype=rewards.dtype)
    return jnp.mean(jnp.sum(discounts * rewards, axis=-1))


def euler_residuals(model, policy, states, mc_shocks):
    """Euler residual at each decision state. ``states`` is ``(n, state_dim)``."""

    def at_state(state):
        control = policy(state)

        def realize(shock):
            nxt = model.transition(state, control, shock)
            return model.expectation(nxt, policy(nxt))

        expectation = jnp.mean(jax.vmap(realize)(mc_shocks), axis=0)
        return model.residuals(state, control, expectation)[0]

    return jax.vmap(at_state)(states)


def evaluate(model, policies, config=None):
    """Welfare and Euler residual for each policy, plus the steady-state rule.

    ``ce_vs_ss`` is the constant consumption change that matches the policy's
    welfare relative to the steady-state saving rule.
    """
    settings = {"eval_seed": 1, "eval_episodes": 32, "eval_periods": 1024, "eval_mc_draws": 32}
    if config:
        settings.update(config)
    states0, shocks, mc_shocks = draw_sample(
        model,
        int(settings["eval_episodes"]),
        int(settings["eval_periods"]),
        int(settings["eval_mc_draws"]),
        settings["eval_seed"],
        settings.get("eval_rng"),
    )
    named = {"steady_state": steady_state_policy(model), **policies}
    results = {}
    baseline = None
    for name, policy in named.items():
        metrics = _metrics(model, policy, states0, shocks, mc_shocks)
        results[name] = metrics
        if name == "steady_state":
            baseline = metrics["welfare"]
    periods = int(settings["eval_periods"])
    for metrics in results.values():
        metrics["ce_vs_ss"] = float(_consumption_equivalent(model, metrics["welfare"], baseline, periods))
    return results


def format_metrics(results):
    """Text table of :func:`evaluate`. ``wall_seconds`` is training time excluding compilation."""
    header = (
        f"{'solution':<16} {'welfare':>12} {'ce_vs_ss_%':>12} "
        f"{'residual_mae':>14} {'residual_mse':>14} {'seconds':>8}"
    )
    lines = [header]
    for name, metrics in results.items():
        seconds = metrics.get("wall_seconds")
        clock = f"{seconds:8.2f}" if seconds is not None else f"{'':>8}"
        lines.append(
            f"{name:<16} {metrics['welfare']:12.6g} {100 * metrics['ce_vs_ss']:12.6g} "
            f"{metrics['residual_mae']:14.6g} {metrics['residual_mse']:14.6g} {clock}"
        )
    return "\n".join(lines)


def _metrics(model, policy, states0, shocks, mc_shocks):
    def score(states0, shocks, mc_shocks):
        states, rewards = simulate(model, policy, states0, shocks)
        flat = states.reshape((-1, model.state_ss.shape[0]))
        residual = euler_residuals(model, policy, flat, mc_shocks)
        return welfare(model, rewards), jnp.mean(jnp.abs(residual)), jnp.mean(residual**2)

    level, mae, mse = jax.jit(score)(states0, shocks, mc_shocks)
    return {
        "welfare": float(level),
        "ce_vs_ss": None,
        "residual_mae": float(mae),
        "residual_mse": float(mse),
    }


def _consumption_equivalent(model, value, baseline, horizon):
    """Constant-consumption gap that matches ``value`` relative to ``baseline``."""
    sigma = model.params.sigma
    weight = jnp.sum(model.discount_rate ** jnp.arange(horizon, dtype=model.state_ss.dtype))

    def level(welfare_level):
        excess = ((1.0 - sigma) * welfare_level / weight) ** (1.0 / (1.0 - sigma))
        return model.c_bar + excess

    return level(value) / level(baseline) - 1.0
