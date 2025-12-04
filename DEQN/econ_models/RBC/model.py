"""RBC model wrapper for DEQN training."""

from jax import numpy as jnp
from jax import random


class Model:
    """RBC model with capital adjustment costs for DEQN training."""

    def __init__(
        self,
        precision=jnp.float32,
        beta=0.96,
        alpha=0.3,
        delta=0.1,
        eps_c=2.0,
        rho=0.9,
        phi=2.0,
        shock_sd=0.02,
        volatility_scale=1.0,
        double_precision=False,
        **kwargs,
    ):
        self.precision = jnp.float64 if double_precision else precision
        self.beta = jnp.array(beta, dtype=self.precision)
        self.alpha = jnp.array(alpha, dtype=self.precision)
        self.delta = jnp.array(delta, dtype=self.precision)
        self.eps_c = jnp.array(eps_c, dtype=self.precision)
        self.rho = jnp.array(rho, dtype=self.precision)
        self.phi = jnp.array(phi, dtype=self.precision)
        self.shock_sd = jnp.array(shock_sd * volatility_scale, dtype=self.precision)

        # Compute steady state
        # From Euler: 1 = beta * (1 - delta + alpha * Y/K)
        # At steady state with A=1: Y = K^alpha, so MPK = alpha * K^(alpha-1)
        # 1/beta = 1 - delta + alpha * K^(alpha-1)
        # K_ss = (alpha / (1/beta - 1 + delta))^(1/(1-alpha))
        k_ss_level = (self.alpha / (1 / self.beta - 1 + self.delta)) ** (1 / (1 - self.alpha))
        i_ss_level = self.delta * k_ss_level

        self.k_ss = jnp.log(k_ss_level)
        self.i_ss = jnp.log(i_ss_level)
        self.a_ss = jnp.array(0.0, dtype=self.precision)

        # Standard interface
        self.state_ss = jnp.array([self.k_ss, self.a_ss], dtype=self.precision)
        self.state_sd = jnp.array([1.0, 1.0], dtype=self.precision)
        self.policies_ss = jnp.array([self.i_ss], dtype=self.precision)
        self.policies_sd = jnp.array([1.0], dtype=self.precision)

        self.dim_states = 2
        self.dim_policies = 1
        self.n_sectors = 1

    def initial_state(self, rng, init_range=0):
        """Get initial state with optional perturbation around steady state."""
        rng_k, rng_a = random.split(rng, 2)

        if init_range > 0:
            k_level = random.uniform(
                rng_k,
                minval=(1 - init_range / 100) * jnp.exp(self.k_ss),
                maxval=(1 + init_range / 100) * jnp.exp(self.k_ss),
                dtype=self.precision,
            )
            a_level = random.uniform(
                rng_a,
                minval=-init_range / 100,
                maxval=init_range / 100,
                dtype=self.precision,
            )
        else:
            k_level = jnp.exp(self.k_ss)
            a_level = jnp.array(0.0, dtype=self.precision)

        state_notnorm = jnp.array([jnp.log(k_level), a_level], dtype=self.precision)
        state = (state_notnorm - self.state_ss) / self.state_sd
        return state

    def step(self, state, policy, shock):
        """Transition to next state given current state, policy, and shock."""
        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[0])
        a = state_notnorm[1]

        # AR(1) productivity process
        a_next = self.rho * a + self.shock_sd * shock[0]

        # Denormalize policy
        policy_notnorm = policy * self.policies_sd + self.policies_ss
        Inv = jnp.exp(policy_notnorm[0])

        # Capital accumulation with adjustment costs
        K_next = (1 - self.delta) * K + Inv - (self.phi / 2) * (Inv / K - self.delta) ** 2 * K

        state_next_notnorm = jnp.array([jnp.log(K_next), a_next])
        state_next = (state_next_notnorm - self.state_ss) / self.state_sd
        return state_next

    def expect_realization(self, state_next, policy_next):
        """Compute the realization of expectation terms for the Euler equation."""
        state_next_notnorm = state_next * self.state_sd + self.state_ss
        K_next = jnp.exp(state_next_notnorm[0])
        A_next = jnp.exp(state_next_notnorm[1])

        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss
        Inv_next = jnp.exp(policy_next_notnorm[0])

        Y_next = A_next * K_next**self.alpha
        C_next = Y_next - Inv_next
        P_next = C_next ** (-1 / self.eps_c)

        # Price of capital with adjustment costs
        Pk_next = P_next / (1 - self.phi * (Inv_next / K_next - self.delta))

        # RHS of Euler equation
        expect_realization = P_next * (self.alpha * Y_next / K_next) + Pk_next * (
            (1 - self.delta) + (self.phi / 2) * (Inv_next**2 / K_next**2 - self.delta**2)
        )
        return expect_realization

    def loss(self, state, expect, policy):
        """Compute Euler equation residual (loss) for given state, expectation, and policy."""
        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[0])
        A = jnp.exp(state_notnorm[1])

        policy_notnorm = policy * self.policies_sd + self.policies_ss
        Inv = jnp.exp(policy_notnorm[0])

        Y = A * K**self.alpha
        C = Y - Inv
        P = C ** (-1 / self.eps_c)

        # Price of capital
        Pk = P / (1 - self.phi * (Inv / K - self.delta))

        # LHS of Euler: Pk = beta * E[...]
        MPK = self.beta * expect
        euler_residual = Pk / MPK - 1

        losses_array = jnp.array([euler_residual])
        mean_loss = jnp.mean(losses_array**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(losses_array))
        min_accuracy = jnp.min(1 - jnp.abs(losses_array))
        mean_accuracies_focs = 1 - jnp.abs(losses_array)
        min_accuracies_focs = 1 - jnp.abs(losses_array)

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def sample_shock(self, rng, n_draws=1):
        """Sample shock realizations."""
        return random.normal(rng, shape=(n_draws,), dtype=self.precision)

    def mc_shocks(self, rng=None, mc_draws=8):
        """Sample Monte Carlo shock realizations for expectation computation."""
        if rng is None:
            rng = random.PRNGKey(0)
        return random.normal(rng, shape=(mc_draws, 1), dtype=self.precision)

    def get_aggregates(self, simul_policies, simul_states):
        """Calculate economic aggregates from simulation."""
        simul_policies = jnp.atleast_2d(simul_policies)
        simul_states = jnp.atleast_2d(simul_states)

        states_notnorm = simul_states * self.state_sd + self.state_ss
        K = jnp.exp(states_notnorm[:, 0])
        A = jnp.exp(states_notnorm[:, 1])

        policies_notnorm = simul_policies * self.policies_sd + self.policies_ss
        Inv = jnp.exp(policies_notnorm[:, 0])

        Y = A * K**self.alpha
        C = Y - Inv

        # Steady state values
        K_ss = jnp.exp(self.k_ss)
        Y_ss = K_ss**self.alpha
        I_ss = jnp.exp(self.i_ss)
        C_ss = Y_ss - I_ss

        # Log deviations from steady state
        aggregates = {
            "C": jnp.log(C / C_ss),
            "K": jnp.log(K / K_ss),
            "I": jnp.log(Inv / I_ss),
            "Y": jnp.log(Y / Y_ss),
            "A": jnp.log(A),
        }
        return aggregates

