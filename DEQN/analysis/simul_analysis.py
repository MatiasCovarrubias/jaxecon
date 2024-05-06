import jax
import scipy
import pandas as pd
from jax import numpy as jnp, lax, random
from scipy.stats import skew, kurtosis

def create_episode_simul_analysis_fn(econ_model, config):

    """ Create a function that simulates an episode of the environment. 
    It differes from basic simulation function in that returns both observations and policies."""

    def sample_epis_obs(train_state, epis_rng):
        "sample obs of an episode"
        init_obs = econ_model.initial_obs(epis_rng, config["init_range"])
        period_rngs = random.split(epis_rng, config["periods_per_epis"])
        def period_step(obs, period_rng):
            policy = train_state.apply_fn(train_state.params, obs)
            period_shock = config["simul_vol_scale"]*econ_model.sample_shock(period_rng)     # Sample next obs
            obs_next = econ_model.step(obs, policy, period_shock)                            # apply period steps.
            return obs_next, (obs_next, policy) # we pass it two times because of the syntax of the lax.scan loop
        _, (epis_obs, policies) = lax.scan(period_step, init_obs, jnp.stack(period_rngs))    # we get the obs_batch
        return epis_obs, policies

    return sample_epis_obs

def create_descstats_fn(econ_model, config):

    """ Create a function that calculates descriptive statistics of the simulation results. """

    def autocorrelation(x, lag):
        x = x - jnp.mean(x)
        x_lag = x[lag:]
        x_orig = x[:-lag]
        acf = jnp.dot(x_orig, x_lag) / jnp.dot(x, x)
        return acf

    def statistic(var):
        
        means = float(jnp.mean(var))
        sd = float(jnp.std(var))
        skewness = skew(var)
        kurt = kurtosis(var)
        percentiles = jnp.quantile(var, jnp.array([0.01, 0.25, 0.5, 0.75, 0.99]))
        autocorrelations = [autocorrelation(var, lag) for lag in range(1, 6)]

        desc_stats = [means, sd, skewness, kurt] + percentiles.tolist()
        stats ={"desc_stats": desc_stats, "autocorrelations": autocorrelations}

        return stats

    def descstat(simul_policies):

        desc_stats = {}
        autocorrs = {}
        aggregates = econ_model.get_aggregates(simul_policies)
        for agg_name, agg_value in aggregates:
            statistics = statistic(agg_value)
            desc_stats[agg_name] = statistics["desc_stats"]
            autocorrs[agg_name] = statistics["autocorrelations"]
        desc_stats = desc_stats.append({' ':  ["Mean", "Sd", "Skewness", "Kurtosis","Q1","Q25","Q50","Q75","Q99"]})
        autocorrs = autocorrs.append({' ': ["Lag 1", "Lag 2", "Lag 3", "Lag 4", "Lag 5"]})
        desc_stats_df = pd.DataFrame(desc_stats)
        desc_stats_df = desc_stats_df.transpose()
        desc_stats_df = desc_stats_df.round(3)
        autocorrs_df = pd.DataFrame(autocorrs)
        autocorrs_df = autocorrs_df.transpose()
        autocorrs_df = autocorrs_df.round(3)

        return desc_stats_df, autocorrs_df
    
    return descstat

