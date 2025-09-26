import jax
from jax import lax, random
from jax import numpy as jnp


def create_stochss_fn(econ_model, config):
    """
    Create stochastic steady state function.

    Note: This function expects inputs from simulation_verbose_fn which returns states
          in log deviation form (not normalized by standard deviations).
    """

    def random_draws(simul_obs, n_draws, seed=0):
        """Sample random points from simulation observations (in logdev form)."""
        n_simul = simul_obs.shape[0]
        key = random.PRNGKey(seed)
        indices = random.choice(key, n_simul, shape=(n_draws,), replace=False)
        obs_draws = simul_obs[indices, :]
        return obs_draws

    def simul_traject_lastobs(econ_model, train_state, shocks, obs_init_logdev):
        """
        Simulate trajectory and return last observation.

        Args:
            obs_init_logdev: Initial observation in log deviation form

        Returns:
            final_obs_logdev: Final observation in log deviation form
        """

        def step(obs_logdev, shock):
            # Convert logdevs to normalized state for neural network and econ_model calls
            obs_normalized = obs_logdev / econ_model.state_sd
            policy_normalized = train_state.apply_fn(train_state.params, obs_normalized)
            next_obs_normalized = econ_model.step(obs_normalized, policy_normalized, shock)

            # Convert back to logdevs for consistency
            next_obs_logdev = next_obs_normalized * econ_model.state_sd
            policy_logdev = policy_normalized * econ_model.policies_sd

            obs_pol_pair = (obs_logdev, policy_logdev)
            return next_obs_logdev, obs_pol_pair

        final_obs_logdev, _ = lax.scan(step, obs_init_logdev, shocks)
        return final_obs_logdev

    def stochss_fn(simul_obs, train_state):
        """
        Compute stochastic steady state.

        Args:
            simul_obs: Simulation observations in log deviation form
            train_state: Trained neural network state

        Returns:
            policy_stoch_ss: Policy at stochastic steady state (normalized form)
            stoch_ss_mean: Mean state at stochastic steady state (logdev form)
            stoch_ss_std: Standard deviation of states (logdev form)
        """
        sample_fromdist = random_draws(simul_obs, config["n_draws"], config["seed"])
        zero_shocks = jnp.zeros(shape=(config["time_to_converge"], econ_model.n_sectors))
        stoch_ss = jax.vmap(simul_traject_lastobs, in_axes=(None, None, None, 0))(
            econ_model, train_state, zero_shocks, sample_fromdist
        )
        stoch_ss_mean_logdev = jnp.mean(stoch_ss, axis=0)
        stoch_ss_std_logdev = jnp.std(stoch_ss, axis=0)

        # Convert logdev mean to normalized state for neural network call
        stoch_ss_mean_normalized = stoch_ss_mean_logdev / econ_model.state_sd
        policy_stoch_ss = train_state.apply_fn(train_state.params, stoch_ss_mean_normalized)

        # aggs_stochss_dict = econ_model.get_aggregates(policy_stoch_ss_logdev)
        # return aggs_stochss_dict
        return policy_stoch_ss, stoch_ss_mean_logdev, stoch_ss_std_logdev

    return stochss_fn
