"""APG step: descend on the negative discounted return."""

import jax
from jax import numpy as jnp
from jax import random

from trainers.policy import init_policy, policy_control, sgd


def train_apg(model, config):
    rng, init_rng = random.split(random.PRNGKey(config["seed"]))
    params = init_policy(init_rng, model, config["hidden"])
    periods = int(config["periods"])
    episodes = int(config["episodes"])
    discounts = model.discount_rate ** jnp.arange(periods, dtype=model.state_ss.dtype)

    def episode_loss(current, key):
        key, state_key = random.split(key)
        state = model.initial_state(state_key)
        shocks = jax.vmap(model.sample_shock)(random.split(key, periods))

        def step(carry, shock):
            control = policy_control(current, carry, model, config["control_width"])
            reward = model.utility(carry, control)
            return model.transition(carry, control, shock), reward

        _, rewards = jax.lax.scan(step, state, shocks)
        return -jnp.sum(discounts * rewards)

    def step(current, key):
        keys = random.split(key, episodes)
        loss, grads = jax.value_and_grad(lambda p: jnp.mean(jax.vmap(lambda k: episode_loss(p, k))(keys)))(current)
        return loss, grads

    n_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    return sgd(step, params, rng, n_steps, config["learning_rate"])
