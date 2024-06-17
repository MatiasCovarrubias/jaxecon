from jax import numpy as jnp
from jax import random
from jax.scipy.stats import norm

class Rbc_intermediates():
    """A JAX implementation of an RBC model."""

    def __init__(self, policies_ss=[1,1], precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, eps_c=0.5, rho=0.9, phi=2, mu=0.5, sigma_q=0.5,  Sigma_A=[[0.01,0],[0,0.01]]):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)
        self.rho = jnp.array(rho, dtype=precision)
        self.phi = jnp.array(phi, dtype=precision)
        self.mu = jnp.array(mu, dtype=precision)
        self.sigma_q = jnp.array(sigma_q, dtype=precision)
        self.Sigma_A = jnp.array(Sigma_A, dtype=precision)

        # set steady state and standard deviations for normalization
        self.policies_ss = jnp.array(policies_ss, dtype=precision)
        self.a_ss = jnp.array([0.0,0.0], dtype=precision)
        self.k_ss = jnp.log(jnp.exp(self.policies_ss) / self.delta)
        self.obs_ss = jnp.concatenate([self.k_ss,self.a_ss])
        self.obs_sd = jnp.array([1, 1, 1, 1], dtype=precision)  # use 1 if you don't have an estimate

        # number of policies
        self.n_actions = len(policies_ss)

        # print a dictionary with all the variables to debug
        # print({
        #     "beta": self.beta,
        #     "alpha": self.alpha,
        #     "delta": self.delta,
        #     "eps_c": self.eps_c,
        #     "rho": self.rho,
        #     "phi": self.phi,
        #     "xi": self.xi,
        #     "sigma_c": self.sigma_c,
        #     "Sigma_A": self.Sigma_A,
        #     "policies_ss": self.policies_ss,
        #     "a_ss": self.a_ss,
        #     "k_ss": self.k_ss,
        #     "obs_ss": self.obs_ss,
        #     "obs_sd": self.obs_sd,
        #     })

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

        # print a dictionary with all the variables to debug
        # print({
        #     "obs_init_notnorm": obs_init_notnorm,
        #     "obs_init": obs_init,
        #     })
        
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

        # print a dictionary with all the local variables to debug
        # print({
        #     "obs_notnorm": obs_notnorm,
        #     "K": K,
        #     "a": a,
        #     "a_tplus1": a_tplus1,
        #     "I": I,
        #     "K_tplus1": K_tplus1,
        #     "obs_next_notnorm": obs_next_notnorm,
        #     "obs_next": obs_next,
        #     })

        return obs_next


    def expect_realization(self, obs_next, policy_next):
        """ A realization (given a shock) of the expectation terms in system of equation """

        obs_next_notnorm = obs_next*self.obs_sd + self.obs_ss# denormalize
        K_next = jnp.exp(obs_next_notnorm[:2]) # put in levels
        A_next = jnp.exp(obs_next_notnorm[2:])
        I_next = policy_next*jnp.exp(self.policies_ss)
        Y_next = A_next*K_next**self.alpha
        M_next = Y_next[1] - I_next[1]
        Q1_next = ( self.mu**(1/self.sigma_q) * Y_next[0]**((self.sigma_q-1)/self.sigma_q) + (1-self.mu)**(1/self.sigma_q) * M_next**((self.sigma_q-1)/self.sigma_q) ) ** (self.sigma_q/(self.sigma_q-1))
        C_next = Q1_next - I_next[0]
        P1_next = (C_next) ** (-self.eps_c ** (-1))
        P2_next = P1_next * ((1-self.mu)*Q1_next/M_next)**(1/self.sigma_q) 
        Pk1_next = P1_next * (1-self.phi*(I_next[0]/K_next[0]-self.delta))**(-1)
        Pk2_next = P2_next * (1-self.phi*(I_next[1]/K_next[1]-self.delta))**(-1)

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization1 = (P1_next*(self.mu*Q1_next/Y_next[0])**(1/self.sigma_q) * (self.alpha*Y_next[0]/K_next[0]) + Pk1_next*((1-self.delta) + self.phi/2*(I_next[0]**2 / K_next[0]**2-self.delta**2)))
        expect_realization2 = (P2_next*(self.alpha*Y_next[1]/K_next[1]) + Pk2_next*((1-self.delta) + self.phi/2*(I_next[1]**2 / K_next[1]**2-self.delta**2)))

        # print a dictionary with all the local variables to debug
        # print({
        #     "obs_next_notnorm": obs_next_notnorm,
        #     "K_next": K_next,
        #     "A_next": A_next,
        #     "I_next": I_next,
        #     "Y_next": Y_next,
        #     "C_next": C_next,
        #     "Cagg_next": Cagg_next,
        #     "P_next": P_next,
        #     "Pk_next": Pk_next,
        #     "expect_realization": expect_realization,
        #     })

        return jnp.array([expect_realization1,expect_realization2])

    def loss(self, obs, expect, policy):
        """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

        obs_notnorm = obs*self.obs_sd + self.obs_ss# denormalize
        K = jnp.exp(obs_notnorm[:2]) # put in levels
        A = jnp.exp(obs_notnorm[2:])
        I = policy*jnp.exp(self.policies_ss)
        Y = A*K**self.alpha
        M = Y[1] - I[1]
        Q1 = ( self.mu**(1/self.sigma_q) * Y[0]**((self.sigma_q-1)/self.sigma_q) + (1-self.mu)**(1/self.sigma_q) * M**((self.sigma_q-1)/self.sigma_q) ) ** (self.sigma_q/(self.sigma_q-1))
        C = Q1 - I[0]
        P1 = (C) ** (-self.eps_c ** (-1))
        P2 = P1 * ((1-self.mu)*Q1/M)**(1/self.sigma_q) 
        Pk1 = P1 * (1-self.phi*(I[0]/K[0]-self.delta))**(-1)
        Pk2 = P2 * (1-self.phi*(I[1]/K[1]-self.delta))**(-1)
        MPK1 = self.beta * expect[0]
        MPK2 = self.beta * expect[1]
        
        K1_loss = Pk1/MPK1 - 1
        K2_loss = Pk2/MPK2 - 1
        K_loss = jnp.array([K1_loss,K2_loss])
        losses_array = jnp.array([K_loss])
        mean_loss = jnp.mean(losses_array**2)
        max_loss = jnp.max(losses_array**2) # here there is just one, but more gemore generally.
        mean_accuracy = jnp.mean(1-jnp.abs(losses_array))
        min_accuracy = jnp.min(1-jnp.abs(losses_array))
        mean_accuracies_foc = 1-jnp.abs(losses_array)
        min_accuracies_foc = 1-jnp.abs(losses_array)
        
        # print a dictionary with all the local variables to debug
        # print({
        #     "obs_notnorm": obs_notnorm,
        #     "K": K,
        #     "A": A,
        #     "I": I,
        #     "Y": Y,
        #     "C": C,
        #     "Cagg": Cagg,
        #     "P": P,
        #     "Pk": Pk,
        #     "MPK": MPK,
        #     "K_loss": K_loss,
        #     "losses_array": losses_array,
        #     "mean_loss": mean_loss,
        #     "max_loss": max_loss,
        #     "mean_accuracy": mean_accuracy,
        #     "min_accuracy": min_accuracy,
        #     "mean_accuracies_foc": mean_accuracies_foc,
        #     "min_accuracies_foc": min_accuracies_foc,
        #     })

        return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accuracies_foc, min_accuracies_foc
    
    def loss_ss(self, policy):
        """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

        I=jnp.exp(policy)
        K = I/self.delta
        Y = K**self.alpha
        M = Y[1] - I[1]
        Q1 = ( self.mu**(1/self.sigma_q) * Y[0]**((self.sigma_q-1)/self.sigma_q) + (1-self.mu)**(1/self.sigma_q) * M**((self.sigma_q-1)/self.sigma_q) ) ** (self.sigma_q/(self.sigma_q-1))
        C = Q1 - I[0]
        MPK1 = self.beta * ((1-self.delta)+(self.mu*Q1/Y[0])**(1/self.sigma_q) * (self.alpha*Y[0]/K[0]))
        MPK2 = self.beta * ((1-self.delta)+(self.alpha*Y[1]/K[1]))
        
        
        K1_loss = 1/MPK1 - 1
        K2_loss = 1/MPK2 - 1
        K_loss = jnp.array([K1_loss,K2_loss])
        losses_array = jnp.array([K_loss])
        mean_loss = jnp.mean(losses_array**2)
        
        return mean_loss
    

    def sample_shock(self, rng):
        """ sample one realization of the shock.
        Uncomment second line for continuous shocks instead of grid """
        shock =  random.multivariate_normal(rng, jnp.zeros((2,)), self.Sigma_A)
        # lower_bound = norm.ppf(0.01)  # 1st percentile (inverse CDF)
        # upper_bound = norm.ppf(0.99)  # 99th percentile

        # # Windsorize along each dimension independently
        # shock_windsorized = jnp.clip(shock, lower_bound, upper_bound)
        return shock

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """ sample omc_draws realizations of the shock (for monte-carlo)
        Uncomment second line for continuous shocks instead of grid """
        # return  jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816])
        shocks =  random.multivariate_normal(rng, jnp.zeros((2,)), self.Sigma_A, shape=(mc_draws,))
        # Calculate quantiles for windsorization
        # lower_bound = norm.ppf(0.01)  # 1st percentile (inverse CDF)
        # upper_bound = norm.ppf(0.99)  # 99th percentile

        # Windsorize along each dimension independently
        # shocks= jnp.clip(shocks, lower_bound, upper_bound)

        return shocks

    def utility(self,C,L):
        U = (1/(1-self.eps_c**(-1)))*C**(1-self.eps_c**(-1))
        return U
    
    def get_aggregates(self, policy, obs):
        """Calculate aggregates from simulation policies"""
        # Ensure inputs are treated as arrays for consistent indexing
        obs_notnorm = obs*self.obs_sd + self.obs_ss# denormalize
        K = jnp.exp(obs_notnorm[:2]) # put in levels
        A = jnp.exp(obs_notnorm[2:])
        I = policy*jnp.exp(self.policies_ss)
        Y = A*K**self.alpha
        C = Y - I
        Cagg = ( (self.xi**(1/self.sigma_c)).T @ C**((self.sigma_c-1)/self.sigma_c) ) ** (self.sigma_c/(self.sigma_c-1))
        P = (Cagg) ** (-self.eps_c ** (-1)) * (Cagg * self.xi / C) ** (1 / self.sigma_c)
        Pk = P * (1-self.phi*(I/K-self.delta))**(-1)
        Yagg = Y@P
        Kagg = K@Pk
        Iagg = I@Pk

        # get steady state
        Kss = jnp.exp(self.k_ss)
        Yss = Kss**self.alpha
        Iss = jnp.exp(self.policies_ss)
        Css = Yss - Iss
        Caggss = ( (self.xi**(1/self.sigma_c)).T @ Css**((self.sigma_c-1)/self.sigma_c) ) ** (self.sigma_c/(self.sigma_c-1))
        Pss = (Caggss) ** (-self.eps_c ** (-1)) * (Caggss * self.xi / Css) ** (1 / self.sigma_c)
        Pkss = Pss
        Yaggss = Yss@Pss
        Kaggss = Kss@Pkss
        Iaggss = Iss@Pkss
        
        # aggregate loggdevs
        Cagg_logdev = jnp.log(Cagg/Caggss)
        K_logdev = jnp.log(Kagg/Kaggss)
        I_logdev = jnp.log(Iagg/Iaggss)
        Y_logdev = jnp.log(Yagg/Yaggss)

        #idiosyncratic logdevs
        C1_logdev = jnp.log(C[0]/Css[0])
        C2_logdev = jnp.log(C[1]/Css[1])
        K1_logdev = jnp.log(K[0]/Kss[0])
        K2_logdev = jnp.log(K[1]/Kss[1])
 
        return jnp.array([Cagg_logdev,K_logdev,I_logdev,Y_logdev,C1_logdev,C2_logdev,K1_logdev,K2_logdev])
        
    def get_aggregates_keys(self):
        return ["Cagg","Kagg","Iagg","Yagg","C1","C2","K1","K2"]
    
    def get_shocks_statistics(self):
        # sample many shocks and gets statistics
        shocks = self.mc_shocks(mc_draws=10000)
        mean_shocks = jnp.mean(shocks, axis=0)
        sd_shocks = jnp.std(shocks, axis=0)
        return mean_shocks, sd_shocks