from jax import numpy as jnp
from jax import random


class RbcCES_SteadyState:
    """A JAX implementation of the steady state of an Rbc model."""

    def __init__(self, precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, sigma_y=0.5, eps_c=2.0, eps_l=0.5):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.sigma_y = jnp.array(sigma_y, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)
        self.eps_l = jnp.array(eps_l, dtype=precision)

    def loss(self, policy):
        """Calculate loss associated with observing obs, having policy_params, and expectation exp"""

        policy_notnorm = policy
        C = policy_notnorm[0]
        L = policy_notnorm[1]
        K = policy_notnorm[2]
        Inv = policy_notnorm[3]
        P = policy_notnorm[4]
        Pk = policy_notnorm[5]
        Y = policy_notnorm[6]
        theta = policy_notnorm[7]

        # Calculate the FOC for Pk
        MgUtC = (C) ** (-self.eps_c ** (-1))
        MPL = ((1 - self.alpha) * Y / L) ** (self.sigma_y ** (-1))
        MPK = self.beta * ((1 - self.delta) + (self.alpha * Y / K) ** (self.sigma_y ** (-1)))
        Ydef = (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))

        C_loss = P / MgUtC - 1
        L_loss = theta * L ** (self.eps_l ** (-1)) / (P * MPL) - 1
        K_loss = 1 / MPK - 1
        Inv_loss = Pk / P - 1
        P_loss = Y / (C + Inv) - 1
        Pk_loss = Inv / (self.delta * K) - 1
        Y_loss = Y / Ydef - 1
        theta_loss = P - 1
        losses_array = jnp.array([C_loss, L_loss, K_loss, Inv_loss, P_loss, Pk_loss, Y_loss, theta_loss])
        mean_loss = jnp.mean(losses_array**2)
        return mean_loss


class RbcCES:
    """A JAX implementation of an RBC model."""

    def __init__(
        self,
        policies_ss=[1, 1, 1, 1, 1, 1, 1],
        precision=jnp.float32,
        theta=2,
        beta=0.96,
        alpha=0.3,
        delta=0.1,
        sigma_y=0.5,
        eps_c=2.0,
        eps_l=0.5,
        rho=0.9,
        phi=2,
        shock_sd=0.02,
    ):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.sigma_y = jnp.array(sigma_y, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)
        self.eps_l = jnp.array(eps_l, dtype=precision)
        self.theta = jnp.array(theta, dtype=precision)
        self.rho = jnp.array(rho, dtype=precision)
        self.phi = jnp.array(phi, dtype=precision)
        self.shock_sd = jnp.array(shock_sd, dtype=precision)

        # set steady state and standard deviations for normalization
        self.policies_ss = jnp.array(policies_ss, dtype=precision)
        self.a_ss = jnp.array(0, dtype=precision)
        self.k_ss = jnp.array(policies_ss[2], dtype=precision)
        self.obs_ss = jnp.array([self.k_ss, 0], dtype=precision)
        self.obs_sd = jnp.array([1, 1], dtype=precision)  # use 1 if you don't have an estimate

        # number of policies
        self.n_actions = len(policies_ss)

    def initial_obs(self, rng, init_range=0):
        """Get initial obs given first shock"""
        rng_k, rng_a = random.split(rng, 2)
        K = random.uniform(
            rng_k,
            minval=(1 - init_range / 100) * jnp.exp(self.k_ss),
            maxval=(1 + init_range / 100) * jnp.exp(self.k_ss),
            dtype=self.precision,
        )  # get uniform draw around the steady state
        A = random.uniform(
            rng_a,
            minval=(1 - init_range / 100) * jnp.exp(self.a_ss),
            maxval=(1 + init_range / 100) * jnp.exp(self.a_ss),
            dtype=self.precision,
        )  # get uniform draw around the steady state

        obs_init_notnorm = jnp.array([jnp.log(K), jnp.log(A)], dtype=self.precision)
        obs_init = (obs_init_notnorm - self.obs_ss) / self.obs_sd  # normalize
        return obs_init

    def step(self, obs, policy, shock):
        """A period step of the model, given current obs, the shock and policy"""

        obs_notnorm = obs * self.obs_sd + self.obs_ss  # denormalize
        K = jnp.exp(obs_notnorm[0])  # Kt in levels
        a = obs_notnorm[1]  # a_{t}
        a_tplus1 = self.rho * a + self.shock_sd * shock[0]  # recover a_{t+1}
        policy_notnorm = policy * jnp.exp(self.policies_ss)  # multiply by stst pols in level
        K_tplus1 = policy_notnorm[2]  # get K_{t+1}
        obs_next_notnorm = jnp.array([jnp.log(K_tplus1), a_tplus1])  # concatenate observation
        obs_next = (obs_next_notnorm - self.obs_ss) / self.obs_sd  # normalize

        return obs_next

    def expect_realization(self, obs_next, policy_next):
        """A realization (given a shock) of the expectation terms in system of equation"""

        obs_next_notnorm = obs_next * self.obs_sd + self.obs_ss  # denormalize
        K_next = jnp.exp(obs_next_notnorm[0])  # put in levels
        A_next = jnp.exp(obs_next_notnorm[1])

        policy_notnorm = policy_next * jnp.exp(self.policies_ss)
        I_next = policy_notnorm[3]
        P_next = policy_notnorm[4]
        Pk_next = policy_notnorm[5]
        Y_next = policy_notnorm[6]

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization = P_next * A_next ** (1 - self.sigma_y ** (-1)) * (self.alpha * Y_next / K_next) ** (
            1 / self.sigma_y
        ) + Pk_next * ((1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2))

        return expect_realization

    def loss(self, obs, expect, policy):
        """Calculate loss associated with observing obs, having policy_params, and expectation exp"""

        obs_notnorm = obs * self.obs_sd + self.obs_ss  # denormalize
        K = jnp.exp(obs_notnorm[0])  # put in levels
        A = jnp.exp(obs_notnorm[1])

        policy_notnorm = policy * jnp.exp(self.policies_ss)
        C = policy_notnorm[0]
        L = policy_notnorm[1]
        K_tplus1 = policy_notnorm[2]
        I = policy_notnorm[3]
        P = policy_notnorm[4]
        Pk = policy_notnorm[5]
        Y = policy_notnorm[6]

        # Calculate the FOC for Pk
        MgUtC = (C) ** (-self.eps_c ** (-1))
        MPL = A ** (1 - self.sigma_y ** (-1)) * ((1 - self.alpha) * Y / L) ** (self.sigma_y ** (-1))
        MPK = self.beta * expect
        Pkmodel = P * (1 - self.phi * (I / K - self.delta)) ** (-1)
        K_tplus1_def = (1 - self.delta) * K + I - (self.phi / 2) * (I / K - self.delta) ** 2 * K
        Ydef = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))

        C_loss = P / MgUtC - 1
        L_loss = self.theta * L ** (self.eps_l ** (-1)) / (P * MPL) - 1
        K_loss = Pk / MPK - 1
        I_loss = Pk / Pkmodel - 1
        P_loss = Y / (C + I) - 1
        Pk_loss = K_tplus1 / K_tplus1_def - 1
        Y_loss = Y / Ydef - 1

        losses_array = jnp.array([C_loss, L_loss, K_loss, I_loss, P_loss, Pk_loss, Y_loss])
        mean_loss = jnp.mean(losses_array**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(losses_array))
        min_accuracy = jnp.min(1 - jnp.abs(losses_array))
        mean_accuracies_foc = 1 - jnp.abs(losses_array)
        min_accuracies_foc = 1 - jnp.abs(losses_array)
        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_foc, min_accuracies_foc

    def sample_shock(self, rng, n_draws=1):
        """sample one realization of the shock.
        Uncomment second line for continuous shocks instead of grid"""
        # return random.choice(rng, jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816]))
        return random.normal(rng, shape=(n_draws,), dtype=self.precision)

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """sample omc_draws realizations of the shock (for monte-carlo)
        Uncomment second line for continuous shocks instead of grid"""
        # return  jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816])
        return random.normal(rng, shape=(mc_draws, 1), dtype=self.precision)

    def utility(self, C, L):
        C_notnorm = C * jnp.exp(self.policies_ss[0])
        L_notnorm = L * jnp.exp(self.policies_ss[1])
        U = (1 / (1 - self.eps_c ** (-1))) * (C_notnorm) ** (1 - self.eps_c ** (-1)) - self.theta * (
            1 / (1 + self.eps_l ** (-1))
        ) * L_notnorm ** (1 + self.eps_l ** (-1))
        return U

    def get_aggregates(self, simul_policies):
        """Calculate aggregates from simulation policies"""
        C = simul_policies[:, 0]
        L = simul_policies[:, 1]
        K = simul_policies[:, 2]
        I = simul_policies[:, 3]
        Y = simul_policies[:, 6]
        aggregates = {"C": C, "L": L, "K": K, "I": I, "Y": Y}
        return aggregates

    def get_aggregates_keys(self):
        """Return the keys for aggregates in order"""
        return ["C", "L", "K", "I", "Y"]
