"""Model-agnostic welfare evaluation of a policy in a ``WelfareEnvironment``.

Two policies evaluated with the same ``rng`` see the same initial states and
the same shock paths, so their welfare difference is a common-random-number
estimate of the consumption-equivalent gain.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random, vmap


class WelfareRollout(NamedTuple):
    welfare: jax.Array
    """Mean discounted return over episodes."""
    welfare_per_episode: jax.Array
    """Discounted return of each episode, shape ``(n_epis,)``."""
    states: jax.Array
    """Visited states, shape ``(n_epis, horizon, state_dim)``."""
    actions: jax.Array
    """Actions taken, shape ``(n_epis, horizon, action_dim)``."""


def create_welfare_fn(env, horizon, n_epis, init_range=0, simul_vol_scale=1.0):
    """Return ``welfare_fn(policy_fn, rng) -> WelfareRollout``.

    ``policy_fn(obs) -> action`` acts on one state. Episodes start from
    ``env.initial_state(rng, init_range)`` and run ``horizon`` periods with
    shocks scaled by ``simul_vol_scale``; there is no terminal continuation.
    """
    horizon = int(horizon)
    if horizon < 1 or n_epis < 1:
        raise ValueError("horizon and n_epis must be positive")

    def episode(policy_fn, rng):
        rng_init, rng_shocks = random.split(rng)
        obs0 = env.initial_state(rng_init, init_range, mode="box")
        shocks = simul_vol_scale * vmap(env.sample_shock)(random.split(rng_shocks, horizon))

        def period(carry, shock):
            obs, welfare, discount = carry
            action = policy_fn(obs)
            reward = env.training_reward(obs, action)
            obs_next = env.transition(obs, action, shock)
            return (obs_next, welfare + discount * reward, discount * env.discount_rate), (obs, action)

        init = (obs0, jnp.zeros((), dtype=obs0.dtype), jnp.ones((), dtype=obs0.dtype))
        (_, welfare, _), (states, actions) = lax.scan(period, init, shocks)
        return welfare, states, actions

    def welfare_fn(policy_fn, rng):
        welfare, states, actions = vmap(lambda key: episode(policy_fn, key))(random.split(rng, n_epis))
        return WelfareRollout(jnp.mean(welfare), welfare, states, actions)

    return welfare_fn
