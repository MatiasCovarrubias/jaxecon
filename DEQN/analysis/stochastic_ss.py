import jax
from jax import numpy as jnp, lax, random

def create_stochss_fn(econ_model, config):

    def random_draws(simul_obs, n_draws, seed=0):
        n_simul = simul_obs.shape[0]
        key = random.PRNGKey(seed)
        indices = random.choice(key, n_simul, shape=(n_draws,), replace=False)
        obs_draws = simul_obs[indices, :]
        return obs_draws

    def simul_traject_lastobs(econ_model, train_state, shocks, obs_init):
        def step(obs, shock):
            policy = train_state.apply_fn(train_state.params, obs)
            next_obs = econ_model.step(obs, policy, shock)
            obs_pol_pair = (obs,policy)
            return next_obs, obs_pol_pair
        final_obs, _ = lax.scan(step, obs_init, shocks)
        return final_obs

    def stochss_fn(simul_obs, train_state, n_draws=1000, seed=0, time_to_converge=300):
        sample_fromdist = random_draws(simul_obs, n_draws, seed)
        zero_shocks = jnp.zeros(shape=(time_to_converge,1))
        stoch_ss = jax.vmap(simul_traject_lastobs, in_axes = (None,None,None,0))(econ_model,train_state,zero_shocks,sample_fromdist)
        stoch_ss = jnp.mean(stoch_ss, axis=0)
        policy_stoch_ss = train_state.apply_fn(train_state.params, stoch_ss)
        policy_stoch_ss_logdev = jnp.log(policy_stoch_ss)
        # aggs_stochss_dict = econ_model.get_aggregates(policy_stoch_ss_logdev)
        # return aggs_stochss_dict
        return policy_stoch_ss_logdev    

    return stochss_fn