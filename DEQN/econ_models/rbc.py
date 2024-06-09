from jax import numpy as jnp
from jax import random

class Rbc_SteadyState():
    """A JAX implementation of the steady state of an Rbc model."""

    def __init__(self, precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, eps_c=2):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)

    def loss(self, policy):
        """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

        I = policy[0]
        K = I/self.delta
        Y = K**self.alpha
        MPK = self.beta * ((1-self.delta)+(self.alpha*Y/K))
        K_loss = 1/MPK - 1

        losses_array = jnp.array([K_loss])
        mean_loss = jnp.mean(losses_array**2)
        return mean_loss
    
class Rbc_capadj():
    """A JAX implementation of an RBC model."""

    def __init__(self, policies_ss=[1], precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, eps_c=2, rho=0.9, phi=2, shock_sd=0.02):
        self.precision = precision
        # set parameters
        self.beta = jnp.array(beta, dtype=precision)
        self.alpha = jnp.array(alpha, dtype=precision)
        self.delta = jnp.array(delta, dtype=precision)
        self.eps_c = jnp.array(eps_c, dtype=precision)
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

    def initial_obs(self, rng, init_range = 0):
        """ Get initial obs given first shock """
        rng_k, rng_a = random.split(rng,2)
        K = random.uniform(
                rng_k, minval=(1-init_range/100) * jnp.exp(self.k_ss), maxval=(1+init_range/100) * jnp.exp(self.k_ss), dtype=self.precision
            )  # get uniform draw around the steady state
        A = random.uniform(
                rng_a, minval=(1-init_range/100) * jnp.exp(self.a_ss), maxval=(1+init_range/100) * jnp.exp(self.a_ss), dtype=self.precision
            )  # get uniform draw around the steady state

        obs_init_notnorm = jnp.array([jnp.log(K), jnp.log(A)], dtype=self.precision)
        obs_init = (obs_init_notnorm-self.obs_ss)/self.obs_sd # normalize
        return obs_init

    def step(self, obs, policy, shock):
        """ A period step of the model, given current obs, the shock and policy """

        obs_notnorm = obs*self.obs_sd + self.obs_ss # denormalize
        K = jnp.exp(obs_notnorm[0])                 # Kt in levels
        a = obs_notnorm[1]                    # a_{t}
        a_tplus1 = self.rho * a + self.shock_sd*shock[0]   # recover a_{t+1}
        I = policy[0]*jnp.exp(self.policies_ss[0])             # multiply by stst pols in level
        K_tplus1 = (1-self.delta)*K + I - (self.phi/2) * (I/K - self.delta)**2 * K 
        obs_next_notnorm = jnp.array([jnp.log(K_tplus1),a_tplus1])  #concatenate observation
        obs_next = (obs_next_notnorm-self.obs_ss)/self.obs_sd        # normalize

        return obs_next


    def expect_realization(self, obs_next, policy_next):
        """ A realization (given a shock) of the expectation terms in system of equation """

        obs_next_notnorm = obs_next*self.obs_sd + self.obs_ss# denormalize
        K_next = jnp.exp(obs_next_notnorm[0]) # put in levels
        A_next = jnp.exp(obs_next_notnorm[1])

        I_next = policy_next[0]*jnp.exp(self.policies_ss[0])
        Y_next = A_next*K_next**self.alpha
        C_next = Y_next - I_next
        P_next = (C_next) ** (-self.eps_c ** (-1))
        Pk_next = P_next * (1-self.phi*(I_next/K_next-self.delta))**(-1)

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization = (P_next*(self.alpha*Y_next/K_next) + Pk_next*((1-self.delta) + self.phi/2*(I_next**2 / K_next**2-self.delta**2)))

        return expect_realization

    def loss(self, obs, expect, policy):
        """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

        obs_notnorm = obs*self.obs_sd + self.obs_ss# denormalize
        K = jnp.exp(obs_notnorm[0]) # put in levels
        A = jnp.exp(obs_notnorm[1])
        I = policy[0]*jnp.exp(self.policies_ss[0])
        Y = A*K**self.alpha
        C = Y - I
        P = (C) ** (-self.eps_c ** (-1))
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
        # return random.choice(rng, jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816]))
        return random.normal(rng, shape=(n_draws,), dtype=self.precision)

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """ sample omc_draws realizations of the shock (for monte-carlo)
        Uncomment second line for continuous shocks instead of grid """
        # return  jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816])
        return random.normal(rng, shape=(mc_draws,1), dtype = self.precision)

    def utility(self,C,L):
        U = (1/(1-self.eps_c**(-1)))*C**(1-self.eps_c**(-1))
        return U
    
    def get_aggregates(self, simul_policies, simul_obs):
        """Calculate aggregates from simulation policies"""
        K = simul_policies[:,0] # put in levels
        A = simul_policies[:,1]
        I = simul_policies[:,0]
        Y = A*K**self.alpha
        C = Y - I
        aggregates = {"C": C, "K": K, "I": I, "Y": Y, "A": A}
        return aggregates
    
class Rbc():
  """A JAX implementation of an RBC model."""

  def __init__(self, precision=jnp.float32):
    self.precision = precision
    # set parameters
    self.beta = jnp.array(0.96, dtype=precision)
    self.alpha = jnp.array(0.3, dtype=precision)
    self.delta = jnp.array(0.1, dtype=precision)
    self.rho = jnp.array(0.9, dtype=precision)
    self.shock_sd = jnp.array(0.02, dtype=precision)


    # set steady state and standard deviations for normalization
    self.k_ss = jnp.log((self.alpha/(1/self.beta-1+self.delta))**(1/(1-self.alpha)))
    self.a_ss = jnp.array(0, dtype=precision)
    self.obs_ss = jnp.array([self.k_ss, 0], dtype=precision)
    self.obs_sd = jnp.array([1, 1], dtype=precision)  # use 1 if you don't have an estimate
    self.policy_ss = self.k_ss

    # number of policies
    self.n_actions = 1

  def initial_obs(self, rng, init_range = 0):
    """ Get initial obs given first shock """
    rng_k, rng_a = random.split(rng,2)
    K = random.uniform(
            rng_k, minval=(1-init_range/100) * jnp.exp(self.k_ss), maxval=(1+init_range/100) * jnp.exp(self.k_ss), dtype=self.precision
        )  # get uniform draw around the steady state
    A = random.uniform(
            rng_a, minval=(1-init_range/100) * jnp.exp(self.a_ss), maxval=(1+init_range/100) * jnp.exp(self.a_ss), dtype=self.precision
        )  # get uniform draw around the steady state

    obs_init_notnorm = jnp.array([jnp.log(K), jnp.log(A)], dtype=self.precision)
    obs_init = (obs_init_notnorm-self.obs_ss)/self.obs_sd # normalize
    return obs_init

  def step(self, obs, policy, shock):
    """ A period step of the model, given current obs, the shock and policy """

    obs_notnorm = obs*self.obs_sd + self.obs_ss # denormalize
    K = jnp.exp(obs_notnorm[0])                 # Kt in levels
    a = obs_notnorm[1]                    # a_{t}
    a_tplus1 = self.rho * a + self.shock_sd*shock[0]   # recover a_{t+1}
    policy_notnorm = policy*jnp.exp(self.policy_ss)             # multiply by stst pols in level
    # K_tplus1 = (1-self.delta)*K + policy_notnorm[0]             #get K_{t+1}
    K_tplus1 = policy_notnorm[0]             #get K_{t+1}
    obs_next_notnorm = jnp.array([jnp.log(K_tplus1),a_tplus1])  #concatenate observation
    obs_next = (obs_next_notnorm-self.obs_ss)/self.obs_sd        # normalize

    return obs_next


  def expect_realization(self, obs_next, policy_next):
    """ A realization (given a shock) of the expectation terms in system of equation """

    policy_notnorm = policy_next*jnp.exp(self.policy_ss) # multiply by stst pols in levels
    K_tplus1 = policy_notnorm[0]                                # define investment

    # Process observation
    obs_notnorm = obs_next*self.obs_sd + self.obs_ss     # denormalize obs
    K = jnp.exp(obs_notnorm[0])                          # K_{t+1} in levels
    a = obs_notnorm[1]                             # a_{t}
    I = K_tplus1 - (1-self.delta)*K
    # Rest of variables
    A = jnp.exp(a)
    Y = A * K**self.alpha
    C = Y-I

    # Calculate the FOC for Pk
    expect_realization = (1/C) * (1+ A * self.alpha * K**(self.alpha-1)-self.delta)

    return expect_realization

  def loss(self, obs, expect, policy):
    """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

    policy_notnorm = policy*jnp.exp(self.policy_ss)
    K_tplus1 = policy_notnorm[0]

    # Process observation
    obs_notnorm = obs*self.obs_sd + self.obs_ss        # denormalize
    K = jnp.exp(obs_notnorm[0])                        # put in levels
    a = obs_notnorm[1]

    # Rest of variables
    I = K_tplus1-(1-self.delta)*K
    A = jnp.exp(a)
    Y = A * K**self.alpha
    C = Y-I

    # Calculate the FOC for Pk
    FOC_loss = (1/C)/(self.beta*expect) - 1
    mean_loss = jnp.mean(jnp.array([FOC_loss**2])) # here there is just one, but more gemore generally.
    max_loss = jnp.max(jnp.array([FOC_loss**2])) # here there is just one, but more gemore generally.
    mean_accuracy = jnp.mean(jnp.array([1-jnp.abs(FOC_loss)]))
    min_accuracy = jnp.min(jnp.array([1-jnp.abs(FOC_loss)]))
    mean_accuracies_foc = jnp.array([1-jnp.abs(FOC_loss)])
    max_accuracies_foc = jnp.array([1-jnp.abs(FOC_loss)])
    return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accuracies_foc, max_accuracies_foc

  def sample_shock(self, rng, n_draws=1):
    """ sample one realization of the shock.
    Uncomment second line for continuous shocks instead of grid """
    # return random.choice(rng, jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816]))
    return random.normal(rng, shape=(n_draws,), dtype=self.precision)

  def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
    """ sample omc_draws realizations of the shock (for monte-carlo)
    Uncomment second line for continuous shocks instead of grid """
    # return  jnp.array([-1.2816,-0.6745,0,0.6745, 1.2816])
    return random.normal(rng, shape=(mc_draws,1), dtype = self.precision)

  def ir_shocks(self):
    """ (Optional) Define a set of shocks sequences that are of interest"""
    # ir_shock_1 = jnp.array([-1]+[0 for i in range(40)])
    # ir_shock_2 = jnp.array([1]+[0 for i in range(40)])
    ir_shock_1 = jnp.zeros(shape=(40,1), dtype = self.precision).at[0,:].set(-1)
    ir_shock_2 = jnp.zeros(shape=(40,1), dtype = self.precision).at[0,:].set(1)

    return jnp.array([ir_shock_1, ir_shock_2])

  def get_econ_stats(self, obs, policy):

    policy_notnorm = policy*jnp.exp(self.policy_ss)
    I = policy_notnorm[0]

    # Process observation
    obs_notnorm = obs*self.obs_sd + self.obs_ss        # denormalize
    K = jnp.exp(obs_notnorm[0])                        # put in levels
    a = obs_notnorm[1]

    # Rest of variables
    A = jnp.exp(a)
    Y = A * K**self.alpha
    C = Y-I

    return jnp.array([K,I,Y,C])