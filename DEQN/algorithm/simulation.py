import jax
from jax import numpy as jnp, lax, random

def create_episode_simul_fn_simple(econ_model, config):

  def sample_epis_obs(train_state, epis_rng):
    "sample obs of an episode"
    init_obs = econ_model.initial_obs(epis_rng, config["init_range"])
    period_rngs = random.split(epis_rng, config["periods_per_epis"])
    def period_step(env_obs, period_rng):
      policy = train_state.apply_fn(train_state.params, env_obs)
      period_shock = config["simul_vol_scale"]*econ_model.sample_shock(period_rng)     # Sample next obs
      obs_next = econ_model.step(env_obs, policy, period_shock)  # apply period steps.
      return obs_next, obs_next # we pass it two times because of the syntax of the lax.scan loop
    _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs)) # we get the obs_batch
    return epis_obs

  return sample_epis_obs

def create_episode_simul_fn_compute_expects(econ_model, config):

  def sample_epis_obs(train_state, epis_rng):
    "sample obs of an episode"
    init_obs = econ_model.initial_obs(epis_rng, config["init_range"])
    period_rngs = random.split(epis_rng, config["periods_per_epis"])
    def period_step(env_obs, period_rng):
      policy, _ = train_state.apply_fn(train_state.params, env_obs)
      period_shock = config["simul_vol_scale"]*econ_model.sample_shock(period_rng)
      obs_next = econ_model.step(env_obs, policy, period_shock)  
      return obs_next, obs_next # we pass it two times because of the syntax of the lax.scan loop
    _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs)) # we get the obs_batch
    return epis_obs

  return sample_epis_obs

def create_episode_simul_fn_proxied(econ_model, config):

  if config["proxy_sampler"]:
    def sample_epis_obs(train_state, epis_rng):
      "sample obs of an episode"
      init_obs = econ_model.initial_obs(epis_rng, config["init_range"])
      period_rngs = random.split(epis_rng, config["periods_per_epis"])
      def period_step(env_obs, period_rng):
        period_shock = config["simul_vol_scale"]*econ_model.sample_shock(period_rng)     # Sample next obs
        obs_next = econ_model.step_loglinear(env_obs, period_shock)                      # apply period steps.
        return obs_next, obs_next # we pass it two times because of the syntax of the lax.scan loop
      _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs)) # we get the obs_batch
      return epis_obs

  else:
    def sample_epis_obs(train_state, epis_rng):
      "sample obs of an episode"
      init_obs = econ_model.initial_obs(epis_rng, config["init_range"])
      period_rngs = random.split(epis_rng, config["periods_per_epis"])
      def period_step(env_obs, period_rng):
        policy = train_state.apply_fn(train_state.params, env_obs)
        period_shock = config["simul_vol_scale"]*econ_model.sample_shock(period_rng)     # Sample next obs
        obs_next = econ_model.step(env_obs, policy, period_shock)  # apply period steps.
        return obs_next, obs_next # we pass it two times because of the syntax of the lax.scan loop
      _, epis_obs = lax.scan(period_step, init_obs, jnp.stack(period_rngs)) # we get the obs_batch
      return epis_obs

  return sample_epis_obs