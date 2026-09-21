"""DEQN step: squared Euler residuals along simulated episodes."""

import jax
from jax import numpy as jnp
from jax import random

from trainers.policy import init_policy, policy_control, sgd


def train_deqn(model, config):
    rng, init_rng = random.split(random.PRNGKey(config["seed"]))
    params = init_policy(init_rng, model, config["hidden"])
    periods = int(config["periods"])
    episodes = int(config["episodes"])
    mc_draws = int(config["mc_draws"])

    def episode_states(current, key):
        key, state_key = random.split(key)
        state = model.initial_state(state_key)
        shocks = jax.vmap(model.sample_shock)(random.split(key, periods))

        def step(carry, shock):
            control = policy_control(current, carry, model, config["control_width"])
            return model.transition(carry, control, shock), carry

        _, states = jax.lax.scan(step, state, shocks)
        return states

    def state_loss(current, state, key):
        control = policy_control(current, state, model, config["control_width"])
        shocks = jax.vmap(model.sample_shock)(random.split(key, mc_draws))

        def realize(shock):
            nxt = model.transition(state, jax.lax.stop_gradient(control), shock)
            nxt_control = policy_control(current, nxt, model, config["control_width"])
            return model.expectation(nxt, nxt_control)

        expectation = jax.lax.stop_gradient(jnp.mean(jax.vmap(realize)(shocks), axis=0))
        residual = model.residuals(state, control, expectation)
        return jnp.mean(residual**2)

    def step(current, key):
        key, episode_key, loss_key = random.split(key, 3)
        states = jax.vmap(lambda episode_key: episode_states(current, episode_key))(
            random.split(episode_key, episodes)
        )
        states = jax.lax.stop_gradient(states.reshape((episodes * periods, states.shape[-1])))
        keys = random.split(loss_key, states.shape[0])

        def batch_loss(current):
            losses = jax.vmap(lambda state, key: state_loss(current, state, key))(states, keys)
            return jnp.mean(losses)

        return jax.value_and_grad(batch_loss)(current)

    n_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    return sgd(step, params, rng, n_steps, config["learning_rate"])
