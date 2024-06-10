from jax import numpy as jnp
from jax import random

class Rbc_twosectors():
    """A JAX implementation of an RBC model."""

    def __init__(self, policies_ss=[1,1], precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, eps_c=2, rho=0.9, phi=2, shock_sd=0.02, xi = [0.5,0.5], sigma_c=0.5, Sigma_A=[[0.01,0],[0,0.01]]):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)
        self.rho = jnp.array(rho, dtype=precision)
        self.phi = jnp.array(phi, dtype=precision)
        self.shock_sd = jnp.array(shock_sd, dtype=precision)
        self.xi = jnp.array(xi, dtype=precision)
        self.sigma_c = jnp.array(sigma_c, dtype=precision)
        self.Sigma_A = jnp.array(Sigma_A, dtype=precision)

        # set steady state and standard deviations for normalization
        self.policies_ss = jnp.array(policies_ss, dtype=precision)
        self.a_ss = jnp.array([0.0,0.0], dtype=precision)
        self.k_ss = jnp.log(jnp.exp(self.policies_ss) / self.delta)
        self.obs_ss = jnp.concatenate([self.k_ss,self.a_ss])
        self.obs_sd = jnp.array([1, 1, 1, 1], dtype=precision)  # use 1 if you don't have an estimate

        # number of policies
        self.n_actions = len(policies_ss)

    def initial_obs(self, rng, init_range = 0):
        """ Get initial obs given first shock """
        rng_k, rng_a = random.split(rng,2)
        K = random.uniform(
                rng_k, shape=(2,), minval=(1-init_range/100) * jnp.exp(self.k_ss), maxval=(1+init_range/100) * jnp.exp(self.k_ss), dtype=self.precision
            )  # get uniform draw around the steady state
        A = random.uniform(
                rng_a, shape=(2,), minval=(1-init_range/100) * jnp.exp(self.a_ss), maxval=(1+init_range/100) * jnp.exp(self.a_ss), dtype=self.precision
            )  # get uniform draw around the steady state

        obs_init_notnorm = jnp.concatenate([jnp.log(K), jnp.log(A)])
        obs_init = (obs_init_notnorm-self.obs_ss)/self.obs_sd # normalize
        return obs_init

    def step(self, obs, policy, shock):
        """ A period step of the model, given current obs, the shock and policy """

        obs_notnorm = obs*self.obs_sd + self.obs_ss # denormalize
        K = jnp.exp(obs_notnorm[:2])                 # Kt in levels
        a = obs_notnorm[2:]                    # a_{t}
        a_tplus1 = self.rho * a + shock   # recover a_{t+1}
        I = policy*jnp.exp(self.policies_ss)             # multiply by stst pols in level
        K_tplus1 = (1-self.delta)*K + I - (self.phi/2) * (I/K - self.delta)**2 * K 
        obs_next_notnorm = jnp.concatenate([jnp.log(K_tplus1),a_tplus1])  #concatenate observation
        obs_next = (obs_next_notnorm-self.obs_ss)/self.obs_sd        # normalize

        return obs_next


    def expect_realization(self, obs_next, policy_next):
        """ A realization (given a shock) of the expectation terms in system of equation """

        obs_next_notnorm = obs_next*self.obs_sd + self.obs_ss# denormalize
        K_next = jnp.exp(obs_next_notnorm[:2]) # put in levels
        A_next = jnp.exp(obs_next_notnorm[2:])
        I_next = policy_next*jnp.exp(self.policies_ss)
        Y_next = A_next*K_next**self.alpha
        C_next = Y_next - I_next
        Cagg_next = ( (self.xi**(1/self.sigma_c)).T @ C_next**((self.sigma_c-1)/self.sigma_c) ) ** (self.sigma_c/(self.sigma_c-1))
        P_next = (Cagg_next) ** (-self.eps_c ** (-1)) * (Cagg_next * self.xi / C_next) ** (1 / self.sigma_c)
        Pk_next = P_next * (1-self.phi*(I_next/K_next-self.delta))**(-1)

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization = (P_next*(self.alpha*Y_next/K_next) + Pk_next*((1-self.delta) + self.phi/2*(I_next**2 / K_next**2-self.delta**2)))

        return expect_realization

    def loss(self, obs, expect, policy):
        """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

        obs_notnorm = obs*self.obs_sd + self.obs_ss# denormalize
        K = jnp.exp(obs_notnorm[:2]) # put in levels
        A = jnp.exp(obs_notnorm[2:])
        I = policy*jnp.exp(self.policies_ss)
        Y = A*K**self.alpha
        C = Y - I
        Cagg = ( (self.xi**(1/self.sigma_c)).T @ C**((self.sigma_c-1)/self.sigma_c) ) ** (self.sigma_c/(self.sigma_c-1))
        P = (Cagg) ** (-self.eps_c ** (-1)) * (Cagg * self.xi / C) ** (1 / self.sigma_c)
        Pk = P * (1-self.phi*(I/K-self.delta))**(-1)
        MPK = self.beta * expect
        
        K_loss = Pk/MPK - 1

        losses_array = jnp.array([K_loss])
        mean_loss = jnp.mean(losses_array**2)
        max_loss = jnp.max(losses_array**2) # here there is just one, but more gemore generally.
        mean_accuracy = jnp.mean(1-jnp.abs(losses_array))
        min_accuracy = jnp.min(1-jnp.abs(losses_array))
        mean_accuracies_foc = 1-jnp.abs(losses_array)
        min_accuracies_foc = 1-jnp.abs(losses_array)
        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accuracies_foc, min_accuracies_foc

    def sample_shock(self, rng, n_draws=1):
        """ sample one realization of the shock.
        Uncomment second line for continuous shocks instead of grid """
        return random.multivariate_normal(rng, jnp.zeros((2,)), self.Sigma_A)

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """ sample omc_draws realizations of the shock (for monte-carlo)
        Uncomment second line for continuous shocks instead of grid """
        # return  jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816])
        return random.multivariate_normal(rng, jnp.zeros((2,)), self.Sigma_A, shape=(mc_draws,))

    def utility(self,C,L):
        U = (1/(1-self.eps_c**(-1)))*C**(1-self.eps_c**(-1))
        return U
    
    def get_aggregates(self, simul_policies, simul_obs):
        """Calculate aggregates from simulation policies"""
        # Ensure inputs are treated as arrays for consistent indexing
        simul_policies = jnp.atleast_2d(simul_policies)
        simul_obs = jnp.atleast_2d(simul_obs)
        Knorm = simul_obs[:,:2]
        Anorm = simul_obs[:,2:]
        K = jnp.exp(Knorm+self.k_ss)
        A = jnp.exp(Anorm)
        I = simul_policies[:,:2]*jnp.exp(self.policies_ss)
        Y = A*K**self.alpha
        C = Y - I
        K_logdev = jnp.log(K/jnp.exp(self.k_ss))
        A_logdev = jnp.log(A)
        Yss = jnp.exp(self.k_ss)**self.alpha
        Iss = jnp.exp(self.policies_ss)
        Css = Yss - Iss
        
        C_logdev = jnp.log(C/Css)
        I_logdev = jnp.log(I/Iss)
        Y_logdev = jnp.log(Y/Yss)

        aggregates = {"C": C_logdev, "K": K_logdev, "I": I_logdev, "Y": Y_logdev, "A": A_logdev}
        return aggregates