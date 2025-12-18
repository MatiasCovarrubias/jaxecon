"""
Three-Equation New Keynesian Model for DEQN training.

Based on Chapter 3 of Galí's "Monetary Policy, Inflation, and the Business Cycle".

The model consists of three equations (Galí notation):

1. New Keynesian Phillips Curve (Eq. 21):
   π_t = β E_t{π_{t+1}} + κ ỹ_t
   where κ ≡ λ(σ + (φ+α)/(1-α))

2. Dynamic IS Equation (Eq. 22):
   ỹ_t = E_t{ỹ_{t+1}} - (1/σ)(i_t - E_t{π_{t+1}} - r^n_t)

3. Natural Rate of Interest (Eq. 23):
   r^n_t = ρ + σ ψ^n_ya E_t{Δa_{t+1}}
        = ρ + σ ψ^n_ya (ρ_a - 1) a_t  [for AR(1) productivity]

4. Interest Rate Rule (Eq. 25):
   i_t = ρ + φ_π π_t + φ_y ỹ_t + v_t

Variables:
- ỹ_t: output gap (denoted 'y_gap' in code, following Galí's tilde-y notation)
- π_t: inflation
- i_t: nominal interest rate
- r^n_t: natural rate of interest
- a_t: productivity shock (AR(1) process)
- v_t: monetary policy shock (AR(1) process)

State variables (exogenous shocks):
- a_t: productivity shock
- v_t: monetary policy shock

Policy/Control variables:
- ỹ_t: output gap
- π_t: inflation

Steady state: ỹ_ss = 0, π_ss = 0 (zero inflation target).
"""

from jax import numpy as jnp
from jax import random


class Model:
    """Three-equation New Keynesian model for DEQN training (Galí Ch. 3)."""

    def __init__(
        self,
        precision=jnp.float32,
        # Preferences
        beta=0.99,  # Discount factor (quarterly)
        sigma=1.0,  # Inverse elasticity of intertemporal substitution (CRRA)
        # Phillips curve slope
        kappa=0.1275,  # Slope of NKPC: κ = λ(σ + (φ+α)/(1-α)), Galí calibration
        # Taylor rule coefficients (Galí notation: φ_π, φ_y)
        phi_pi=1.5,  # Taylor rule coefficient on inflation
        phi_y=0.5 / 4,  # Taylor rule coefficient on output gap (Galí: 0.5/4 quarterly)
        # Shock processes
        rho_a=0.9,  # Persistence of productivity shock
        rho_v=0.5,  # Persistence of monetary policy shock
        sigma_a=0.01,  # Std dev of productivity innovation
        sigma_v=0.0025,  # Std dev of monetary policy innovation (25 basis points)
        # Natural rate coefficient
        psi_ya=1.0,  # Elasticity of natural output to productivity (ψ^n_ya)
        # Scaling
        volatility_scale=1.0,
        double_precision=False,
        **kwargs,
    ):
        self.precision = jnp.float64 if double_precision else precision

        # Store parameters
        self.beta = jnp.array(beta, dtype=self.precision)
        self.sigma = jnp.array(sigma, dtype=self.precision)
        self.kappa = jnp.array(kappa, dtype=self.precision)
        self.phi_pi = jnp.array(phi_pi, dtype=self.precision)
        self.phi_y = jnp.array(phi_y, dtype=self.precision)
        self.rho_a = jnp.array(rho_a, dtype=self.precision)
        self.rho_v = jnp.array(rho_v, dtype=self.precision)
        self.sigma_a = jnp.array(sigma_a * volatility_scale, dtype=self.precision)
        self.sigma_v = jnp.array(sigma_v * volatility_scale, dtype=self.precision)
        self.psi_ya = jnp.array(psi_ya, dtype=self.precision)

        # Natural rate coefficient (Galí Eq. 23):
        # r^n_t = ρ + σ ψ^n_ya E_t{Δa_{t+1}}
        # For AR(1) process a_t = ρ_a a_{t-1} + ε_t:
        #   E_t{a_{t+1}} = ρ_a a_t
        #   E_t{Δa_{t+1}} = E_t{a_{t+1} - a_t} = (ρ_a - 1) a_t
        # So: r^n_t - ρ = σ ψ^n_ya (ρ_a - 1) a_t
        # In deviations from steady state:
        self.r_rn = self.sigma * self.psi_ya * (self.rho_a - 1)

        # Steady state (in deviations from ρ): all zeros
        self.y_gap_ss = jnp.array(0.0, dtype=self.precision)  # ỹ_ss = 0
        self.pi_ss = jnp.array(0.0, dtype=self.precision)  # π_ss = 0
        self.a_ss = jnp.array(0.0, dtype=self.precision)  # a_ss = 0
        self.v_ss = jnp.array(0.0, dtype=self.precision)  # v_ss = 0

        # Standard interface
        # States: [a_t, v_t] (exogenous shocks)
        self.state_ss = jnp.array([self.a_ss, self.v_ss], dtype=self.precision)
        self.state_sd = jnp.array([1.0, 1.0], dtype=self.precision)

        # Policies: [ỹ_t, π_t] (output gap and inflation)
        self.policies_ss = jnp.array([self.y_gap_ss, self.pi_ss], dtype=self.precision)
        self.policies_sd = jnp.array([1.0, 1.0], dtype=self.precision)

        self.dim_states = 2
        self.dim_policies = 2
        self.n_sectors = 1

    def initial_state(self, rng, init_range=0):
        """Get initial state with optional perturbation around steady state."""
        rng_a, rng_v = random.split(rng, 2)

        if init_range > 0:
            a = random.uniform(
                rng_a,
                minval=-init_range / 100,
                maxval=init_range / 100,
                dtype=self.precision,
            )
            v = random.uniform(
                rng_v,
                minval=-init_range / 100,
                maxval=init_range / 100,
                dtype=self.precision,
            )
        else:
            a = jnp.array(0.0, dtype=self.precision)
            v = jnp.array(0.0, dtype=self.precision)

        state_notnorm = jnp.array([a, v], dtype=self.precision)
        state = (state_notnorm - self.state_ss) / self.state_sd
        return state

    def step(self, state, policy, shock):
        """Transition to next state given current state, policy, and shock.

        The NK model is purely forward-looking with no endogenous states.
        States evolve as exogenous AR(1) processes:
            a_{t+1} = ρ_a a_t + ε^a_{t+1}
            v_{t+1} = ρ_v v_t + ε^v_{t+1}
        """
        state_notnorm = state * self.state_sd + self.state_ss
        a = state_notnorm[0]  # Productivity shock
        v = state_notnorm[1]  # Monetary policy shock

        # AR(1) shock processes
        a_next = self.rho_a * a + self.sigma_a * shock[0]
        v_next = self.rho_v * v + self.sigma_v * shock[1]

        state_next_notnorm = jnp.array([a_next, v_next], dtype=self.precision)
        state_next = (state_next_notnorm - self.state_ss) / self.state_sd
        return state_next

    def expect_realization(self, state_next, policy_next):
        """Compute the realization of expectation terms for the Euler equations.

        Returns the terms needed for computing expectations:
        - E_t{ỹ_{t+1}}: expected output gap
        - E_t{π_{t+1}}: expected inflation
        """
        # Denormalize next period policy
        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss
        y_gap_next = policy_next_notnorm[0]  # ỹ_{t+1}
        pi_next = policy_next_notnorm[1]  # π_{t+1}

        # Return terms needed for expectations: [E{ỹ_{t+1}}, E{π_{t+1}}]
        expect_realization = jnp.array([y_gap_next, pi_next], dtype=self.precision)
        return expect_realization

    def loss(self, state, expect, policy):
        """Compute Euler equation residuals for given state, expectation, and policy.

        Euler equations (from Galí Ch. 3):

        1. Dynamic IS (Eq. 22):
           ỹ_t = E_t{ỹ_{t+1}} - (1/σ)(i_t - E_t{π_{t+1}} - r^n_t)

        2. NKPC (Eq. 21):
           π_t = β E_t{π_{t+1}} + κ ỹ_t

        Interest rate from Taylor Rule (Eq. 25):
           i_t = φ_π π_t + φ_y ỹ_t + v_t  (in deviations from ρ)
        """
        # Denormalize state
        state_notnorm = state * self.state_sd + self.state_ss
        a = state_notnorm[0]  # Productivity shock
        v = state_notnorm[1]  # Monetary policy shock

        # Denormalize policy
        policy_notnorm = policy * self.policies_sd + self.policies_ss
        y_gap = policy_notnorm[0]  # Output gap ỹ_t
        pi = policy_notnorm[1]  # Inflation π_t

        # Expectations
        E_y_gap_next = expect[0]  # E_t{ỹ_{t+1}}
        E_pi_next = expect[1]  # E_t{π_{t+1}}

        # Natural rate of interest (deviation from ρ, Eq. 23)
        r_n = self.r_rn * a

        # Taylor rule: interest rate (deviation from ρ, Eq. 25)
        # i_t - ρ = φ_π π_t + φ_y ỹ_t + v_t
        i = self.phi_pi * pi + self.phi_y * y_gap + v

        # Dynamic IS residual (Eq. 22):
        # ỹ_t - E_t{ỹ_{t+1}} + (1/σ)(i_t - E_t{π_{t+1}} - r^n_t) = 0
        is_residual = y_gap - E_y_gap_next + (1 / self.sigma) * (i - E_pi_next - r_n)

        # NKPC residual (Eq. 21):
        # π_t - β E_t{π_{t+1}} - κ ỹ_t = 0
        nkpc_residual = pi - self.beta * E_pi_next - self.kappa * y_gap

        losses_array = jnp.array([is_residual, nkpc_residual])
        mean_loss = jnp.mean(losses_array**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(losses_array))
        min_accuracy = jnp.min(1 - jnp.abs(losses_array))
        mean_accuracies_focs = 1 - jnp.abs(losses_array)
        min_accuracies_focs = 1 - jnp.abs(losses_array)

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def sample_shock(self, rng, n_draws=1):
        """Sample shock realizations (2 shocks: productivity and monetary policy)."""
        return random.normal(rng, shape=(n_draws, 2), dtype=self.precision).squeeze()

    def mc_shocks(self, rng=None, mc_draws=8):
        """Sample Monte Carlo shock realizations for expectation computation."""
        if rng is None:
            rng = random.PRNGKey(0)
        return random.normal(rng, shape=(mc_draws, 2), dtype=self.precision)

    def get_aggregates(self, simul_policies, simul_states):
        """Calculate economic aggregates from simulation."""
        simul_policies = jnp.atleast_2d(simul_policies)
        simul_states = jnp.atleast_2d(simul_states)

        # Denormalize states
        states_notnorm = simul_states * self.state_sd + self.state_ss
        a = states_notnorm[:, 0]  # Productivity shock
        v = states_notnorm[:, 1]  # Monetary policy shock

        # Denormalize policies
        policies_notnorm = simul_policies * self.policies_sd + self.policies_ss
        y_gap = policies_notnorm[:, 0]  # Output gap ỹ_t
        pi = policies_notnorm[:, 1]  # Inflation π_t

        # Compute interest rate from Taylor rule (Eq. 25)
        i = self.phi_pi * pi + self.phi_y * y_gap + v

        # Natural rate of interest (Eq. 23)
        r_n = self.r_rn * a

        # Real interest rate (ex-ante): r_t = i_t - E_t{π_{t+1}}
        # Approximate with: r_t ≈ i_t - π_t
        r = i - pi

        aggregates = {
            "y_gap": y_gap,  # Output gap (ỹ_t)
            "pi": pi,  # Inflation (π_t)
            "i": i,  # Nominal interest rate (i_t)
            "r": r,  # Real interest rate (approx)
            "r_n": r_n,  # Natural rate of interest (r^n_t)
            "a": a,  # Productivity shock (a_t)
            "v": v,  # Monetary policy shock (v_t)
        }
        return aggregates

    def get_taylor_rule_interest(self, y_gap, pi, v=0.0):
        """Compute interest rate from Taylor rule (Eq. 25)."""
        return self.phi_pi * pi + self.phi_y * y_gap + v
