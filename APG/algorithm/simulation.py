from typing import NamedTuple
from jax import numpy as jnp, random
import jax


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


def create_simul_episode_fn(env, periods_per_epis):

    def simul_episode(params, train_state, epis_rng):
        obs, env_state = env.reset(epis_rng)
        period_rngs = random.split(epis_rng, periods_per_epis)
        # epis_rng, *period_rngs = random.split(epis_rng, periods_per_epis + 1)
        runner_state = params, env_state, obs, 0, 1

        def period_step(runner_state, period_rng):
            params, env_state, obs, returns, discount = runner_state
            # SELECT ACTION
            action, value_notnorm = train_state.apply_fn(params, obs)
            value = value_notnorm * env.value_ss

            # STEP ENV
            obs, env_state, reward, done, info = env.step(period_rng, env_state, action)
            transition = Transition(done, action, value, reward, obs, info)
            returns = returns + discount * reward
            discount = env.discount_rate * discount
            runner_state = (params, env_state, obs, returns, discount)
            return runner_state, transition

        # GET TRAJECTORIES
        runner_state, trajectory = jax.lax.scan(period_step, runner_state, jnp.stack(period_rngs))

        # CALCULATE DISCOUNTED RETURN AND LAST VALUE
        _, _, last_obs, returns, discount = runner_state
        _, last_val_notnorm = train_state.apply_fn(train_state.params, last_obs)
        last_val = last_val_notnorm * env.value_ss
        returns = returns + discount * last_val

        return returns, trajectory, last_val

    return simul_episode
