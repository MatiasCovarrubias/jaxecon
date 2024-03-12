from jax import numpy as jnp
from jax import random

class RbcCES_SteadyState():
  """A JAX implementation of an RBC model."""

  def __init__(self, precision=jnp.float32, beta=0.96, alpha=0.3, delta=0.1, sigma_y=0.5, eps_c=2, eps_l=0.5):
    self.precision = precision
    # set parameters
    self.beta = jnp.array(beta, dtype=precision)
    self.alpha = jnp.array(alpha, dtype=precision)
    self.delta = jnp.array(delta, dtype=precision)
    self.sigma_y = jnp.array(sigma_y, dtype=precision)
    self.eps_c = jnp.array(eps_c, dtype=precision)
    self.eps_l= jnp.array(eps_l, dtype=precision)

  def loss(self, policy):
    """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

    policy_notnorm = policy
    C = policy_notnorm[0]
    L = policy_notnorm[1]
    K = policy_notnorm[2]
    I = policy_notnorm[3]
    P = policy_notnorm[4]
    Pk = policy_notnorm[5]
    Y = policy_notnorm[6]
    theta = policy_notnorm[7]

    # Calculate the FOC for Pk
    MgUtC = (C - theta * 1 / (1 + self.eps_l ** (-1)) * L ** (1 + self.eps_l ** (-1))) ** (-self.eps_c ** (-1))
    MPL = ((1 - self.alpha) * Y / L)**(self.sigma_y ** (-1))
    MPK = self.beta * ((1-self.delta)+(self.alpha*Y/K)**(self.sigma_y ** (-1)))
    Ydef = (self.alpha**(1/self.sigma_y) * K**((self.sigma_y-1)/self.sigma_y) + (1-self.alpha)**(1/self.sigma_y) * L**((self.sigma_y-1)/self.sigma_y) ) ** (self.sigma_y/(self.sigma_y-1))

    C_loss = P/MgUtC - 1
    L_loss = theta*L**(self.eps_l ** (-1)) / MPL -1
    K_loss = 1/MPK - 1
    I_loss = I/(self.delta*K) - 1
    P_loss = Y/(C+self.delta*K) - 1
    Pk_loss = Pk/P -1 
    Y_loss = Y/Ydef-1 
    theta_loss = P-1
    losses_array = jnp.array([C_loss,L_loss,K_loss,I_loss,P_loss,Pk_loss, Y_loss,theta_loss])
    mean_loss = jnp.mean(losses_array**2)
    return mean_loss

  
class RbcCES():
  """A JAX implementation of an RBC model."""

  def __init__(self, precision=jnp.float32, policies_ss=[3.8758656e-01, -1.2288518e-01,  2.4382138e-01,  2.3841855e-07, 4.7065210e-01], theta = jnp.exp(7.1949315e-01)):
    self.precision = precision
    # set parameters
    self.beta = jnp.array(0.96, dtype=precision)
    self.alpha = jnp.array(0.3, dtype=precision)
    self.delta = jnp.array(0.1, dtype=precision)
    self.rho = jnp.array(0.9, dtype=precision)
    self.shock_sd = jnp.array(0.02, dtype=precision)
    self.sigma_y = jnp.array(0.5, dtype=precision)
    self.phi = jnp.array(2, dtype=precision)
    self.eps_c = jnp.array(2, dtype=precision)
    self.eps_l= jnp.array(0.5, dtype=precision)
    self.theta = jnp.array(theta, dtype=precision)

    # set steady state and standard deviations for normalization
    self.a_ss = jnp.array(0, dtype=precision)
    self.policy_ss = jnp.array(policies_ss, dtype=precision)
    self.k_ss = jnp.array(policies_ss[2], dtype=precision)
    self.obs_ss = jnp.array([self.k_ss, 0], dtype=precision)
    self.obs_sd = jnp.array([1, 1], dtype=precision)  # use 1 if you don't have an estimate

    # number of policies
    self.n_actions = 5

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
    K_tplus1 = policy_notnorm[2]             #get K_{t+1}
    obs_next_notnorm = jnp.array([jnp.log(K_tplus1),a_tplus1])  #concatenate observation
    obs_next = (obs_next_notnorm-self.obs_ss)/self.obs_sd        # normalize

    return obs_next


  def expect_realization(self, obs_next, policy_next):
    """ A realization (given a shock) of the expectation terms in system of equation """

    policy_notnorm = policy_next*jnp.exp(self.policy_ss) # multiply by stst pols in levels
    C = policy_notnorm[0]
    L = policy_notnorm[1]
    K_tplus1 = policy_notnorm[2]
    P = policy_notnorm[3]
    Y = policy_notnorm[4]

    # Process observation
    obs_notnorm = obs_next*self.obs_sd + self.obs_ss     # denormalize obs
    K = jnp.exp(obs_notnorm[0])                          # K_{t+1} in levels
    a = obs_notnorm[1]   
    
    # DEefine rest of the variables
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