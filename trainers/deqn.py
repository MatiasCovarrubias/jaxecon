"""DEQN step: squared Euler residuals, with the expectation stopped."""

import jax
from jax import numpy as jnp
from jax import random

from trainers.policy import init_policy, policy_control, sgd


def train_deqn(model, config):
    rng, init_rng = random.split(random.PRNGKey(config["seed"]))
    params = init_policy(init_rng, model, config["hidden"])
    batch_size = int(config["batch_size"])
    mc_draws = int(config["mc_draws"])

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
        key, state_key, loss_key = random.split(key, 3)
        states = jax.vmap(model.initial_state)(random.split(state_key, batch_size))
        keys = random.split(loss_key, batch_size)
        def batch_loss(current):
            losses = jax.vmap(lambda state, key: state_loss(current, state, key))(states, keys)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(batch_loss)(current)
        return loss, grads

    n_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    return sgd(step, params, rng, n_steps, config["learning_rate"])
