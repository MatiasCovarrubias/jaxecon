"""DEQN step on the Euler gap, continuation held fixed.

One shock per period, on the same paths as APG. The loss is the mean
discounted moment ``(dK'/dz) * (q - beta * E) * z``. ``z`` is the live logit.
The weight is stopped, so the step moves only the current policy. The gap
``q - beta * E`` is the shared residual times ``beta * E``.
"""

import jax
from jax import numpy as jnp

from trainers.evaluate import path
from trainers.policy import adam, batch_draw, init_policy, policy_control, policy_logit, saving_rate, split_streams


def train_deqn(model, config):
    streams = split_streams(config["seed"])
    params = init_policy(streams["init"], model, config["hidden"])
    periods = int(config["periods"])
    episodes = int(config["episodes"])
    antithetic = bool(config["antithetic"])
    path_count = episodes * (2 if antithetic else 1)
    discounts = model.discount_rate ** jnp.arange(periods, dtype=model.state_ss.dtype)

    def rollout(current, state, shocks):
        frozen = jax.lax.stop_gradient(current)
        _, states, _ = path(model, lambda carry: policy_control(frozen, carry, model), state, shocks)
        return states

    def moment(current, state, shock, discount):
        state = jax.lax.stop_gradient(state)
        shock = jax.lax.stop_gradient(shock)
        logit = policy_logit(current, state, model)
        held = jax.lax.stop_gradient(logit)
        saving = saving_rate(held, model)[None]

        def next_capital(logit_value):
            control = saving_rate(logit_value, model)[None]
            return jnp.exp(model.transition(state, control, shock)[0])

        jacobian = jax.grad(next_capital)(held)
        nxt = jax.lax.stop_gradient(model.transition(state, saving, shock))
        next_saving = saving_rate(policy_logit(jax.lax.stop_gradient(current), nxt, model), model)[None]
        continuation = model.expectation(nxt, next_saving)[0]
        residual = model.residuals(state, saving, continuation[None])[0]
        gap = residual * model.discount_rate * continuation
        return discount * jax.lax.stop_gradient(jacobian * gap) * logit

    def episode_moment(current, state, shocks):
        shock_paths = (shocks, -shocks) if antithetic else (shocks,)
        total = jnp.asarray(0.0, dtype=model.state_ss.dtype)
        for shocks_path in shock_paths:
            states = rollout(current, state, shocks_path)
            total = total + jnp.sum(
                jax.vmap(moment, in_axes=(None, 0, 0, 0))(current, states, shocks_path, discounts)
            )
        return total

    def step(current, key):
        states, shocks = batch_draw(model, key, episodes, periods)

        def batch_loss(current):
            moments = jax.vmap(episode_moment, in_axes=(None, 0, 0))(current, states, shocks)
            return jnp.sum(moments) / path_count

        return jax.value_and_grad(batch_loss)(current)

    n_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    params, metrics = adam(
        step, params, streams["train"], n_steps, config["learning_rate"], config["cosine_alpha"]
    )

    def policy(state):
        return policy_control(params, state, model)

    return metrics, policy, params
