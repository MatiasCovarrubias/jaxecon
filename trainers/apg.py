"""APG step: ascend the discounted return, plus a steady-saving tail.

The tail is ``beta**T`` times ``tail_periods`` of utility at the steady saving
rate, with shocks held at zero. It uses ``utility`` and ``transition``; the
model does not grow an APG method.
"""

import jax
from jax import numpy as jnp

from trainers.evaluate import path
from trainers.policy import adam, batch_draw, init_policy, policy_control, split_streams


def steady_saving_tail(model, state, horizon):
    """Discounted utility of ``horizon`` periods at the steady saving rate."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    control = model.control_ss
    shock = jnp.zeros((1,), dtype=state.dtype)

    def period(carry, _):
        current, discount, value = carry
        reward = model.utility(current, control)
        nxt = model.transition(current, control, shock)
        return (nxt, discount * model.discount_rate, value + discount * reward), None

    init = (
        state,
        jnp.ones((), dtype=state.dtype),
        jnp.zeros((), dtype=state.dtype),
    )
    (_, _, value), _ = jax.lax.scan(period, init, None, length=horizon)
    return value


def train_apg(model, config):
    streams = split_streams(config["seed"])
    params = init_policy(streams["init"], model, config["hidden"])
    periods = int(config["periods"])
    episodes = int(config["episodes"])
    antithetic = bool(config["antithetic"])
    tail_periods = int(config["tail_periods"])
    discounts = model.discount_rate ** jnp.arange(periods, dtype=model.state_ss.dtype)
    tail_discount = model.discount_rate ** periods

    def path_return(current, state, shocks):
        final, _, rewards = path(model, lambda carry: policy_control(current, carry, model), state, shocks)
        total = jnp.sum(discounts * rewards)
        if tail_periods < 1:
            return total
        return total + tail_discount * steady_saving_tail(model, final, tail_periods)

    def episode_loss(current, state, shocks):
        plus = path_return(current, state, shocks)
        if not antithetic:
            return -plus
        return -0.5 * (plus + path_return(current, state, -shocks))

    def step(current, key):
        states, shocks = batch_draw(model, key, episodes, periods)
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean(jax.vmap(episode_loss, in_axes=(None, 0, 0))(p, states, shocks))
        )(current)
        return loss, grads

    n_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    params, metrics = adam(
        step, params, streams["train"], n_steps, config["learning_rate"], config["cosine_alpha"]
    )

    def policy(state):
        return policy_control(params, state, model)

    return metrics, policy, params
