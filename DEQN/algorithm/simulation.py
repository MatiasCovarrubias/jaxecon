from jax import lax, random
from jax import numpy as jnp


def create_episode_simul_fn(econ_model, config):
    """Create a function that simulates an episode of the environment. It returns the observations of the episode."""

    def sample_epis_obs(train_state, epis_rng):
        "sample obs of an episode"
        init_obs = econ_model.initial_state(epis_rng, config["init_range"])
        period_rngs = random.split(epis_rng, config["periods_per_epis"])

        def period_step(env_obs, period_rng):
            policy = train_state.apply_fn(train_state.params, env_obs)
            period_shock = config["simul_vol_scale"] * econ_model.sample_shock(period_rng)  # Sample next obs
            obs_next = econ_model.step(env_obs, policy, period_shock)  # apply period steps.
            return obs_next, obs_next  # we pass it two times because of the syntax of the lax.scan loop

        _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs))  # we get the obs_batch
        return epis_obs

    return sample_epis_obs


def create_episode_simulation_fn_verbose(econ_model, config):
    """Create simulation function that returns both observations and policies in logdevs."""

    def sample_epis_obs_and_policies(train_state, epis_rng):
        """Sample observations and policies for an episode."""
        init_obs = econ_model.initial_state(epis_rng, config["init_range"])
        period_rngs = random.split(epis_rng, config["periods_per_epis"])

        def period_step(env_obs, period_rng):
            policy = train_state.apply_fn(train_state.params, env_obs)
            period_shock = config["simul_vol_scale"] * econ_model.sample_shock(period_rng)
            obs_next = econ_model.step(env_obs, policy, period_shock)
            obs_next_logdev = obs_next * econ_model.state_sd
            policy_logdev = policy * econ_model.policies_sd
            return obs_next, (obs_next_logdev, policy_logdev)

        _, (epis_obs, epis_policies) = lax.scan(period_step, init_obs, jnp.stack(period_rngs))
        return epis_obs, epis_policies

    return sample_epis_obs_and_policies


def create_episode_simul_fn_proxied(econ_model, config):
    """Create a function that simulates an episode of the environment. It returns the observations of the episode.
    It differs from the basic simul fn in that it provides the option of using a proxy sampler."""

    if config["proxy_sampler"]:

        def sample_epis_obs_proxied(epis_rng):
            "sample obs of an episode"
            init_obs = econ_model.initial_state(epis_rng, config["init_range"])
            period_rngs = random.split(epis_rng, config["periods_per_epis"])

            def period_step(env_obs, period_rng):
                period_shock = config["simul_vol_scale"] * econ_model.sample_shock(period_rng)  # Sample next obs
                obs_next = econ_model.step_loglinear(env_obs, period_shock)  # apply period steps.
                return obs_next, obs_next  # we pass it two times because of the syntax of the lax.scan loop

            _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs))  # we get the obs_batch
            return epis_obs

        return sample_epis_obs_proxied

    else:

        def sample_epis_obs_regular(train_state, epis_rng):
            "sample obs of an episode"
            init_obs = econ_model.initial_state(epis_rng, config["init_range"])
            period_rngs = random.split(epis_rng, config["periods_per_epis"])

            def period_step(env_obs, period_rng):
                policy = train_state.apply_fn(train_state.params, env_obs)
                period_shock = config["simul_vol_scale"] * econ_model.sample_shock(period_rng)  # Sample next obs
                obs_next = econ_model.step(env_obs, policy, period_shock)  # apply period steps.
                return obs_next, obs_next  # we pass it two times because of the syntax of the lax.scan loop

            _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs))  # we get the obs_batch
            return epis_obs

        return sample_epis_obs_regular
