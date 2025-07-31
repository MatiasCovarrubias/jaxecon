from jax import numpy as jnp, random


class RbcMultiSector:
    """A JAX implementation of a multi-sector RBC model."""

    def __init__(
        self,
        N=2,
        beta=0.96,
        alpha_values=0.3,
        delta_values=0.1,
        rho_values=0.9,
        shock_sd=0.1,
        xi_values=None,
        sigma_c=0.5,
        discount_rate=0.9,
    ):

        self.N = N
        self.beta = beta
        self.alpha = jnp.ones(N) * alpha_values
        self.delta = jnp.ones(N) * delta_values
        self.rho = jnp.ones(N) * rho_values
        self.discount_rate = discount_rate
        self.shock_sd = jnp.ones(N) * shock_sd
        self.sigma_c = sigma_c
        self.xi = jnp.ones(N) / N if xi_values is None else xi_values

        # Calculate steady state values
        self.k_ss = jnp.log((self.alpha / (1 / self.beta - 1 + self.delta)) ** (1 / (1 - self.alpha)))
        self.a_ss = jnp.zeros(N)
        self.obs_ss = jnp.concatenate([self.k_ss, self.a_ss])
        self.obs_sd = jnp.ones(2 * N)
        self.policy_ss = jnp.log(self.delta * jnp.exp(self.k_ss))

        # Steady state rewards and value
        self.I_ss = jnp.exp(self.policy_ss)
        self.K_ss = jnp.exp(self.k_ss)
        self.A_ss = jnp.exp(self.a_ss)
        self.Y_ss = self.A_ss * self.K_ss**self.alpha
        self.C_ss = self.Y_ss - jnp.exp(self.policy_ss)
        self.Cagg_ss = jnp.sum(self.xi ** (self.sigma_c ** (-1)) * self.C_ss ** (1 - self.sigma_c ** (-1))) ** (
            1 / (1 - self.sigma_c ** (-1))
        )
        self.reward_ss = jnp.log(self.Cagg_ss)
        self.value_ss = self.reward_ss / (1 - self.beta)

        # Utility variables
        self.obs_dim = 2 * N
        self.state_dim = 2 * N
        self.action_dim = N

    def reset(self, rng):
        """Get initial obs given first shock"""
        K = random.uniform(rng, shape=(self.N,), minval=0.95 * jnp.exp(self.k_ss), maxval=1.05 * jnp.exp(self.k_ss))
        A = random.uniform(rng, shape=(self.N,), minval=0.95 * jnp.exp(self.a_ss), maxval=1.05 * jnp.exp(self.a_ss))

        obs_init_notnorm = jnp.concatenate([jnp.log(K), jnp.log(A)])
        obs_init = (obs_init_notnorm - self.obs_ss) / self.obs_sd  # normalize
        state_init = obs_init
        return obs_init, state_init

    def step(self, rng, state, action):
        """A period step of the model, given current obs and policy"""

        # Process observation
        obs = state
        obs_notnorm = obs * self.obs_sd + self.obs_ss  # denormalize
        K = jnp.exp(obs_notnorm[: self.N])
        a = obs_notnorm[self.N :]

        # Evolution of state
        a_tplus1 = self.rho * a + self.shock_sd * random.normal(rng, (self.N,))
        Inv = action * self.I_ss
        K_tplus1 = (1 - self.delta) * K + Inv

        # New observation and state
        new_obs_notnorm = jnp.concatenate([jnp.log(K_tplus1), a_tplus1])
        new_obs = (new_obs_notnorm - self.obs_ss) / self.obs_sd  # normalize
        new_state = new_obs

        # Reward
        A = jnp.exp(a)
        Y = A * K**self.alpha
        C = Y - Inv
        Cagg = jnp.sum(self.xi ** (self.sigma_c ** (-1)) * C ** (1 - self.sigma_c ** (-1))) ** (
            1 / (1 - self.sigma_c ** (-1))
        )
        reward = jnp.log(Cagg)

        # Done, Info
        done = jnp.array(False)
        info = jnp.array([0.0])

        return new_obs, new_state, reward, done, info
