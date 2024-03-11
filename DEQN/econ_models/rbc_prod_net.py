from jax import numpy as jnp
from jax import random
import jax

class RbcProdNet():
  """A JAX implementation of an RBC model."""

  def __init__(self,
    modparams, k_ss, policies_ss, states_sd,
    shocks_sd, policies_sd, A, B, C, D, precision=jnp.float32):

    self.precision = precision
    self.alpha = jnp.array(modparams["paralpha"], dtype = self.precision)
    self.beta = jnp.array(modparams["parbeta"], dtype = self.precision)
    self.delta = jnp.array(modparams["pardelta"], dtype = self.precision)
    self.rho = jnp.array(modparams["parrho"], dtype = self.precision)
    self.eps_c = jnp.array(modparams["pareps_c"], dtype = self.precision)
    self.eps_l = jnp.array(modparams["pareps_l"], dtype = self.precision)
    self.phi = jnp.array(modparams["parphi"], dtype = self.precision)
    self.theta = jnp.array(modparams["partheta"], dtype = self.precision)
    self.sigma_c = jnp.array(modparams["parsigma_c"], dtype = self.precision)
    self.sigma_m = jnp.array(modparams["parsigma_m"], dtype = self.precision)
    self.sigma_q = jnp.array(modparams["parsigma_q"], dtype = self.precision)
    self.sigma_y = jnp.array(modparams["parsigma_y"], dtype = self.precision)
    self.sigma_I = jnp.array(modparams["parsigma_I"], dtype = self.precision)
    self.sigma_l = jnp.array(modparams["parsigma_l"], dtype = self.precision)
    self.xi = jnp.array(modparams["parxi"], dtype = self.precision)
    self.mu = jnp.array(modparams["parmu"], dtype = self.precision)
    self.Gamma_M = jnp.array(modparams["parGamma_M"], dtype = self.precision)
    self.Gamma_I = jnp.array(modparams["parGamma_I"], dtype = self.precision)
    self.Sigma_A = jnp.array(modparams["parSigma_A"], dtype = self.precision)
    self.n_sectors = modparams["parn_sectors"]
    self.obs_ss = jnp.concatenate([k_ss,jnp.zeros(shape=(2*self.n_sectors,), dtype = self.precision)])
    self.state_ss = jnp.concatenate([k_ss,jnp.zeros(shape=(1*self.n_sectors,), dtype = self.precision)])
    self.policies_ss = jnp.array(policies_ss, dtype = self.precision)

    self.A = A
    self.B = B
    self.C = C
    self.D = D
    self.obs_sd = jnp.concatenate([states_sd,shocks_sd])
    self.shocks_sd = jnp.array(shocks_sd, dtype = self.precision)
    self.states_sd = jnp.array(states_sd, dtype = self.precision)
    self.dim_policies = len(policies_ss)
    self.n_actions = len(policies_ss)
    self.dim_obs = len(self.obs_ss)
    self.dim_shock = len(shocks_sd)

  def initial_obs(self, rng, range=1):
    """ Get initial obs given first shock """

    rng_k, rng_a, rng_e, rng_c = random.split(rng,4)
    e = self.sample_shock(rng)                                                  # sample a realization of the shock
    k_ss = self.obs_ss[:self.n_sectors]                                         # get log K in StSt
    a_ss = self.obs_ss[self.n_sectors:2*self.n_sectors]                         # get log A in StSt
    # K_init = random.uniform(rng_k, shape=(self.n_sectors,), minval=(1-range/100)*jnp.exp(self.obs_ss[:self.n_sectors]),maxval = (1+range/100)*jnp.exp(self.obs_ss[:self.n_sectors]))
    K_init = random.uniform(rng_k, shape=(self.n_sectors,), minval=(1-range/100)*jnp.exp(self.obs_ss[:self.n_sectors]),maxval = (1+range/300)*jnp.exp(self.obs_ss[:self.n_sectors]))
    A_init = random.uniform(rng_a, shape=(self.n_sectors,), minval=(1-range/100),maxval = (1+range/100))
    obs_init_notnorm = jnp.concatenate([jnp.log(K_init),jnp.log(A_init),e])
    obs_init = (obs_init_notnorm-self.obs_ss)/self.obs_sd                       # normalize
    return random.choice(rng_c,jnp.array([obs_init,jnp.zeros_like(self.obs_ss)]))
    # return self.obs_ss

  def step(self, obs, policy, shock):
    """ A period step of the model, given current obs, the shock and policy_params """

    obs_notnorm = obs*self.obs_sd + self.obs_ss                                 # denormalize obs
    K = jnp.exp(obs_notnorm[:self.n_sectors])                                   # extract k and put in levels
    a_tmin1 = obs_notnorm[self.n_sectors:2*self.n_sectors]                      # extract a_tmin1
    shock_tmin1 = obs_notnorm[2*self.n_sectors:]                                # extract shock
    a = self.rho * a_tmin1 + shock_tmin1                                        # update a_t
    policy_notnorm = policy*jnp.exp(self.policies_ss)                           # denormalize policy

    I = policy_notnorm[6*self.n_sectors:7*self.n_sectors]
    # print("Inv:", I)
    K_tplus1 = (1-self.delta)*K + I - (self.phi/2) * (I/K - self.delta)**2 * K     # update K
    # K_tplus1 = jnp.where(K_tplus1<0.5*jnp.exp(self.state_ss[:self.n_sectors]),jnp.exp(self.state_ss[:self.n_sectors]),K_tplus1)
    # print("K_tplus1:", K_tplus1)
    # K_tplus1 = policy_notnorm[6*self.n_sectors:7*self.n_sectors]
    obs_next_notnorm = jnp.concatenate([jnp.log(K_tplus1),a, shock])            # calculate next obs not notrm
    obs_next = (obs_next_notnorm-self.obs_ss)/self.obs_sd                       # normalize

    return obs_next

  def step_loglinear(self, obs, shock):
    obs_dev = obs*self.obs_sd
    state_dev = obs_dev[:2*self.n_sectors]
    shock_tmin1 = obs_dev[2*self.n_sectors:]
    new_state_dev =jnp.dot(self.A,state_dev)+jnp.dot(self.B,shock_tmin1)
    new_state = new_state_dev/self.states_sd
    shock_norm = shock/self.shocks_sd
    obs_next = jnp.concatenate([new_state[:2*self.n_sectors], shock_norm])
    return obs_next

  def policy_loglinear(self, obs):
    obs_dev = obs*self.obs_sd
    state_dev = obs_dev[:2*self.n_sectors]
    shock_tmin1 = obs_dev[2*self.n_sectors:]
    policy_devs = jnp.dot(self.C,state_dev)+jnp.dot(self.D,shock_tmin1)
    policy_norm = jnp.exp(policy_devs)
    # state_notnorm = state_dev+self.states_ss
    # K = jnp.exp(state_notnorm[:self.n_sectors])
    # new_state_dev =jnp.dot(self.A,state_dev)+jnp.dot(self.B,shock_tmin1)
    # new_state_notnorm = new_state_dev/self.states_sd+ self.states_ss
    # K_tplus1 = jnp.exp(new_state_notnorm[:self.n_sectors])
    # I_implied = jnp.where(K_tplus1 - (1-self.delta)*K>0,K_tplus1 - (1-self.delta)*K,0.00001)
    # idevs = jnp.exp(jnp.log(I_implied) -self.policies_ss[:self.n_sectors])
    # idevs = jnp.where(idevs<3,idevs,3)
    # idevs = jnp.where(idevs>0.05,idevs,0.05)
    return policy_norm

  def expect_realization(self, obs_next, policy_next):
    """ A realization (given a shock) of the expectation terms in system of equation """

    # Process observation
    obs_next_notnorm = obs_next*self.obs_sd + self.obs_ss# denormalize
    K_next = jnp.exp(obs_next_notnorm[:self.n_sectors]) # put in levels
    a = obs_next_notnorm[self.n_sectors:2*self.n_sectors]
    shock = obs_next_notnorm[2*self.n_sectors:]
    a_next = self.rho * a + shock # recover a_t
    A_next = jnp.exp(a_next)

    # Calculate tplus1 policies
    policy_next_notnorm = policy_next*jnp.exp(self.policies_ss)
    Pk_next = policy_next_notnorm[2*self.n_sectors:3*self.n_sectors]
    I_next = policy_next_notnorm[6*self.n_sectors:7*self.n_sectors]
    # Ktplus1_next = policy_next_notnorm[6*self.n_sectors:7*self.n_sectors]
    P_next = policy_next_notnorm[8*self.n_sectors:9*self.n_sectors]
    Q_next = policy_next_notnorm[9*self.n_sectors:10*self.n_sectors]
    Y_next = policy_next_notnorm[10*self.n_sectors:11*self.n_sectors]

    # solve for I
    # I_next = Ktplus1_next - (1-self.delta)*K_next

    # Solve for the expectation term in the FOC for Ktplus1
    expect_realization = (P_next*A_next**((self.sigma_y-1)/self.sigma_y) * (self.mu*Q_next/Y_next)**(1/self.sigma_q) *(self.alpha*Y_next/K_next)**(1/self.sigma_y)
      + Pk_next*((1-self.delta) + self.phi/2*(I_next**2 / K_next**2-self.delta**2)))

    return jax.lax.stop_gradient(expect_realization)

  def loss(self, obs, expect, policy):
    """ Calculate loss associated with observing obs, having policy_params, and expectation exp """

    # Process observation
    obs_notnorm = obs*self.obs_sd + self.obs_ss# denormalize
    K = jnp.exp(obs_notnorm[:self.n_sectors]) # put in levels
    a_tmin1 = obs_notnorm[self.n_sectors:2*self.n_sectors]
    shock = obs_notnorm[2*self.n_sectors:]
    a = self.rho * a_tmin1 + shock # recover a_t
    A = jnp.exp(a)

    # Process policy
    policy_notnorm = policy*jnp.exp(self.policies_ss)
    C = policy_notnorm[:self.n_sectors]
    L = policy_notnorm[self.n_sectors:2*self.n_sectors]
    Pk = policy_notnorm[2*self.n_sectors:3*self.n_sectors]
    Pm = policy_notnorm[3*self.n_sectors:4*self.n_sectors]
    M = policy_notnorm[4*self.n_sectors:5*self.n_sectors]
    Mout = policy_notnorm[5*self.n_sectors:6*self.n_sectors]
    I = policy_notnorm[6*self.n_sectors:7*self.n_sectors]
    # Ktplus1 = policy_notnorm[6*self.n_sectors:7*self.n_sectors]
    Iout = policy_notnorm[7*self.n_sectors:8*self.n_sectors]
    P = policy_notnorm[8*self.n_sectors:9*self.n_sectors]
    Q = policy_notnorm[9*self.n_sectors:10*self.n_sectors]
    Y = policy_notnorm[10*self.n_sectors:11*self.n_sectors]
    Cagg = policy_notnorm[11*self.n_sectors]
    Lagg = policy_notnorm[11*self.n_sectors+1]
    Yagg = policy_notnorm[11*self.n_sectors+2]
    Iagg = policy_notnorm[11*self.n_sectors+3]
    Magg = policy_notnorm[11*self.n_sectors+4]

    # solve for I or Ktplus1
    # Ktplus1 = (1-self.delta)*K + I - self.phi/2 * (I/K - self.delta)**2 * K
    # Ktplus1 = jnp.where(Ktplus1<0,0.0001,Ktplus1)
    # I =Ktplus1-(1-self.delta)*K

    # get steady state prices to aggregate Y, I and M
    Pss = jnp.exp(self.policies_ss[8*self.n_sectors:9*self.n_sectors])
    Pkss = jnp.exp(self.policies_ss[2*self.n_sectors:3*self.n_sectors])
    Pmss = jnp.exp(self.policies_ss[3*self.n_sectors:4*self.n_sectors])
    capadj_term = 1-self.phi*(I/K-self.delta)
    # print("cap_adj_term:", capadj_term)
    # capadj_term = jnp.where(capadj_term<0,0.0001,capadj_term)
    # auctialiry variables
    Pagg = (self.xi.T @ P ** (1 - self.sigma_c)) ** (1 / (1 - self.sigma_c))
    MgUtCagg = (Cagg - self.theta * 1 / (1 + self.eps_l ** (-1)) * Lagg ** (1 + self.eps_l ** (-1))) ** (-self.eps_c ** (-1))

    # key variables for loss function
    MgUtCmod = MgUtCagg * (Cagg * self.xi / C) ** (1 / self.sigma_c)
    MgUtLmod = MgUtCagg * self.theta * Lagg ** (self.eps_l ** -1) * (L / Lagg) ** (1 / self.sigma_l)
    MPLmod = P * A**((self.sigma_y-1)/self.sigma_y) * (self.mu * Q / Y) ** (1 / self.sigma_q) * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
    MPKmod = self.beta * expect
    Pmdef = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
    Mmod = (1 - self.mu) * (Pm / P) ** (-self.sigma_q) * Q
    Moutmod = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
    Pkdef = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term**(-1)
    Ioutmod = P ** (-self.sigma_I) * jnp.dot( self.Gamma_I,Pk**self.sigma_I * I * capadj_term**(self.sigma_I) )
    Qrc = C + Mout + Iout
    Qdef = ( self.mu**(1/self.sigma_q) * Y**((self.sigma_q-1)/self.sigma_q) + (1-self.mu)**(1/self.sigma_q) * M**((self.sigma_q-1)/self.sigma_q) ) ** (self.sigma_q/(self.sigma_q-1))
    Ydef = A * ( self.alpha**(1/self.sigma_y) * K**((self.sigma_y-1)/self.sigma_y) + (1-self.alpha)**(1/self.sigma_y) * L**((self.sigma_y-1)/self.sigma_y) ) ** (self.sigma_y/(self.sigma_y-1))
    Caggdef = ( (self.xi**(1/self.sigma_c)).T @ C**((self.sigma_c-1)/self.sigma_c) ) ** (self.sigma_c/(self.sigma_c-1))
    Laggdef = jnp.sum( L**((self.sigma_l+1)/self.sigma_l) ) ** (self.sigma_l/(self.sigma_l+1))
    Yaggdef = jnp.sum(Y * Pss)
    Iaggdef = jnp.sum(I * Pkss)
    Maggdef = jnp.sum(M * Pmss)

    C_loss = P/MgUtCmod - 1;
    L_loss = MgUtLmod/MPLmod - 1;
    K_loss = Pk/MPKmod - 1;
    Pm_loss = Pm/Pmdef - 1;
    M_loss = M/Mmod - 1;
    Mout_loss = Mout/Moutmod - 1;
    Pk_loss = Pk/Pkdef - 1;
    Iout_loss = Iout/Ioutmod - 1;
    Qrc_loss = Q/Qrc - 1;
    Qdef_loss = Q/Qdef - 1;
    Ydef_loss = Y/Ydef - 1;
    Caggdef_loss = jnp.array([Cagg/Caggdef - 1]);
    Laggdef_loss = jnp.array([Lagg/Laggdef - 1]);
    Yaggdef_loss = jnp.array([Yagg/Yaggdef - 1]);
    Iaggdef_loss = jnp.array([Iagg/Iaggdef - 1]);
    Maggdef_loss = jnp.array([Magg/Maggdef - 1]);

    losses_array = jnp.concatenate([C_loss,L_loss,K_loss,Pm_loss,M_loss,Mout_loss,Pk_loss,
                              Iout_loss,Qrc_loss,Qdef_loss,Ydef_loss,Caggdef_loss,
                              Laggdef_loss,Yaggdef_loss,Iaggdef_loss,Maggdef_loss], axis =0)

    # jax.debug.print("Policy: {}", policy)
    # jax.debug.print("M: {}", M)
    # Calculate aggregate losses and metrics
    mean_loss = jnp.mean(losses_array**2)
    max_loss = jnp.max(losses_array**2)
    mean_accuracy = jnp.mean(1-jnp.abs(losses_array))
    min_accuracy = jnp.min(1-jnp.abs(losses_array))
    mean_accuracies_focs = jnp.array([jnp.mean(1-jnp.abs(C_loss)),jnp.mean(1-jnp.abs(L_loss)),jnp.mean(1-jnp.abs(K_loss)),jnp.mean(1-jnp.abs(Pm_loss)),jnp.mean(1-jnp.abs(M_loss)),jnp.mean(1-jnp.abs(Mout_loss)),jnp.mean(1-jnp.abs(Pk_loss)),
                              jnp.mean(1-jnp.abs(Iout_loss)),jnp.mean(1-jnp.abs(Qrc_loss)),jnp.mean(1-jnp.abs(Qdef_loss)),jnp.mean(1-jnp.abs(Ydef_loss)),jnp.mean(1-jnp.abs(Caggdef_loss)),
                              jnp.mean(1-jnp.abs(Laggdef_loss)),jnp.mean(1-jnp.abs(Yaggdef_loss)),jnp.mean(1-jnp.abs(Iaggdef_loss)),jnp.mean(1-jnp.abs(Maggdef_loss))])

    min_accuracies_focs = jnp.array([jnp.min(1-jnp.abs(C_loss)),jnp.min(1-jnp.abs(L_loss)),jnp.min(1-jnp.abs(K_loss)),jnp.min(1-jnp.abs(Pm_loss)),jnp.min(1-jnp.abs(M_loss)),jnp.min(1-jnp.abs(Mout_loss)),jnp.min(1-jnp.abs(Pk_loss)),
                              jnp.min(1-jnp.abs(Iout_loss)),jnp.min(1-jnp.abs(Qrc_loss)),jnp.min(1-jnp.abs(Qdef_loss)),jnp.min(1-jnp.abs(Ydef_loss)),jnp.min(1-jnp.abs(Caggdef_loss)),
                              jnp.min(1-jnp.abs(Laggdef_loss)),jnp.min(1-jnp.abs(Yaggdef_loss)),jnp.min(1-jnp.abs(Iaggdef_loss)),jnp.min(1-jnp.abs(Maggdef_loss))])

    return mean_loss, max_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

  def sample_shock(self, rng):
    """ sample one realization of the shock """
    return 1.5*jax.random.multivariate_normal(rng, jnp.zeros((self.n_sectors,)), self.Sigma_A)

  def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
    """ sample mc_draws realizations of the shock (for monte-carlo) """
    return 1.5*jax.random.multivariate_normal(rng, jnp.zeros((self.n_sectors,)), self.Sigma_A, shape=(mc_draws,))

  def ir_shocks(self):
    """ (Optional) Define a set of shocks sequences that are of interest"""
    # ir_shock_1 = jnp.array([-1]+[0 for i in range(40)])
    # ir_shock_2 = jnp.array([1]+[0 for i in range(40)])
    ir_shock_1 = jnp.zeros(shape=(40,self.n_sectors), dtype = self.precision).at[0,0].set(-1)
    ir_shock_2 = jnp.zeros(shape=(40,self.n_sectors), dtype = self.precision).at[0,0].set(1)

    return jnp.array([ir_shock_1, ir_shock_2])