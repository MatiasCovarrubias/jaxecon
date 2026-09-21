"""Score policies on one common sample of states and shocks."""

import jax
from jax import numpy as jnp
from jax import random


def compare(model, policies, config=None):
    """Residual and welfare of each policy on the same episodes.

    Welfare is the mean discounted utility. ``ce_vs_ss`` turns that utility into
    the constant consumption change, relative to the steady-state saving rule,
    that delivers the same utility. Residuals are the model's relative Euler
    residuals on the states visited by that steady-state rule, using one shared
    Monte Carlo shock bank per state.
    """
    settings = {"eval_seed": 1, "eval_episodes": 32, "eval_periods": 32, "eval_mc_draws": 8}
    if config:
        settings.update(config)
    episodes = int(settings["eval_episodes"])
    periods = int(settings["eval_periods"])
    mc_draws = int(settings["eval_mc_draws"])
    dtype = model.state_ss.dtype
    discounts = model.discount_rate ** jnp.arange(periods, dtype=dtype)

    rng = random.PRNGKey(int(settings["eval_seed"]))
    rng, state_key, shock_key, mc_key = random.split(rng, 4)
    states0 = jax.vmap(model.initial_state)(random.split(state_key, episodes))
    shocks = jax.vmap(lambda key: jax.vmap(model.sample_shock)(random.split(key, periods)))(
        random.split(shock_key, episodes)
    )
    mc_shocks = jax.vmap(model.sample_shock)(random.split(mc_key, mc_draws))

    def steady_policy(state):
        return model.control_ss

    named = {"steady_state": steady_policy, **policies}

    def steady_path_states(state, episode_shocks):
        def step(carry, shock):
            return model.transition(carry, model.control_ss, shock), carry

        _, visited = jax.lax.scan(step, state, episode_shocks)
        return visited

    visited = jax.jit(jax.vmap(steady_path_states))(states0, shocks).reshape((-1, model.state_ss.shape[0]))
    baseline = None
    results = {}
    for name, policy in named.items():
        welfare, residual = _score(model, policy, states0, shocks, visited, mc_shocks, discounts)
        results[name] = {
            "welfare": float(welfare),
            "ce_vs_ss": None,
            "residual_mae": float(jnp.mean(jnp.abs(residual))),
            "residual_mse": float(jnp.mean(residual**2)),
        }
        if name == "steady_state":
            baseline = results[name]["welfare"]
    for metrics in results.values():
        metrics["ce_vs_ss"] = float(_consumption_equivalent(model, metrics["welfare"], baseline, periods))
    return results


def _score(model, policy, states0, shocks, visited, mc_shocks, discounts):
    def episode_welfare(state, episode_shocks):
        def step(carry, shock):
            control = policy(carry)
            reward = model.utility(carry, control)
            return model.transition(carry, control, shock), reward

        _, rewards = jax.lax.scan(step, state, episode_shocks)
        return jnp.sum(discounts * rewards)

    def euler(state):
        control = policy(state)

        def realize(shock):
            nxt = model.transition(state, control, shock)
            return model.expectation(nxt, policy(nxt))

        expectation = jnp.mean(jax.vmap(realize)(mc_shocks), axis=0)
        return model.residuals(state, control, expectation)

    welfare = jax.jit(jax.vmap(episode_welfare))(states0, shocks)
    residual = jax.jit(jax.vmap(euler))(visited)
    return jnp.mean(welfare), residual


def format_comparison(results):
    """Text table of :func:`compare`."""
    header = f"{'solution':<16} {'welfare':>12} {'ce_vs_ss_%':>12} {'residual_mae':>14} {'residual_mse':>14}"
    lines = [header]
    for name, metrics in results.items():
        lines.append(
            f"{name:<16} {metrics['welfare']:12.6g} {100 * metrics['ce_vs_ss']:12.6g} "
            f"{metrics['residual_mae']:14.6g} {metrics['residual_mse']:14.6g}"
        )
    return "\n".join(lines)


def _consumption_equivalent(model, welfare, baseline, horizon):
    """Constant-consumption gap that matches ``welfare`` relative to ``baseline``."""
    sigma = model.params.sigma
    weight = jnp.sum(model.discount_rate ** jnp.arange(horizon, dtype=model.state_ss.dtype))

    def level(value):
        excess = ((1.0 - sigma) * value / weight) ** (1.0 / (1.0 - sigma))
        return model.c_bar + excess

    return level(welfare) / level(baseline) - 1.0
