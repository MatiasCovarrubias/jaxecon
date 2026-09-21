"""Economic primitives, dynamics, and diagnostics shared by RBC policy models."""

from abc import ABC, abstractmethod

import numpy as np
from jax import lax, random, vmap
from jax import numpy as jnp


class RbcModelBase(ABC):
    """Common RBC economy without a policy parameterization."""

    def __init__(
        self,
        n_sectors=1,
        precision=jnp.float32,
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        eps_c=0.5,
        rho=0.9,
        rho_values=None,
        phi=2.0,
        i_min_frac=0.0,
        shock_sd=0.02,
        sigma_c=0.5,
        xi_values=None,
        volatility_scale=1.0,
        double_precision=False,
        discount_rate=None,
        cbar_frac=0.0,
    ):
        if not 0 <= i_min_frac < 1:
            raise ValueError("i_min_frac must satisfy 0 <= i_min_frac < 1")
        if not 0 <= cbar_frac < 1:
            raise ValueError("cbar_frac must satisfy 0 <= cbar_frac < 1")
        if rho_values is not None and rho != 0.9:
            raise ValueError("specify only one of rho and rho_values")

        self.precision = jnp.float64 if double_precision else precision
        self.n_sectors = int(n_sectors)
        n = self.n_sectors
        self.beta = jnp.array(beta, dtype=self.precision)
        self.alpha = jnp.ones(n, dtype=self.precision) * jnp.array(alpha, dtype=self.precision)
        self.delta = jnp.ones(n, dtype=self.precision) * jnp.array(delta, dtype=self.precision)
        self.eps_c = jnp.array(eps_c, dtype=self.precision)
        self.risk_aversion = 1 / self.eps_c
        rho_parameter = rho if rho_values is None else rho_values
        self.rho = jnp.ones(n, dtype=self.precision) * jnp.array(rho_parameter, dtype=self.precision)
        self.rho_values = self.rho
        self.phi = jnp.array(phi, dtype=self.precision)
        self.i_min_frac = jnp.array(i_min_frac, dtype=self.precision)
        self.irreversible = float(i_min_frac) > 0
        self.shock_sd = jnp.ones(n, dtype=self.precision) * jnp.array(
            shock_sd * volatility_scale, dtype=self.precision
        )
        self.sigma_c = jnp.array(sigma_c, dtype=self.precision)
        self.xi = (
            jnp.ones(n, dtype=self.precision) / n
            if xi_values is None
            else jnp.array(xi_values, dtype=self.precision)
        )
        self.discount_rate = (
            self.beta
            if discount_rate is None
            else jnp.array(discount_rate, dtype=self.precision)
        )

        k_ss_level = (
            self.alpha / (1 / self.beta - 1 + self.delta)
        ) ** (1 / (1 - self.alpha))
        self.k_ss = jnp.log(k_ss_level)
        self.a_ss = jnp.zeros(n, dtype=self.precision)
        self.state_ss = jnp.concatenate([self.k_ss, self.a_ss])
        self.state_sd = jnp.ones(2 * n, dtype=self.precision)

        self.K_ss = jnp.exp(self.k_ss)
        self.A_ss = jnp.exp(self.a_ss)
        self.Y_ss = self.A_ss * self.K_ss**self.alpha
        self.I_ss = self.delta * self.K_ss
        self.I_min = self.i_min_frac * self.I_ss
        self.C_ss = self.Y_ss - self.I_ss
        self.s_ss = self.I_ss / self.Y_ss
        self.i_ss = jnp.log(self.I_ss)

        self.dim_states = 2 * n
        self.dim_saving = n
        self.obs_ss = self.state_ss
        self.obs_sd = self.state_sd
        self.obs_dim = self.dim_states
        self.state_dim = self.dim_states
        self.N = n

        self.Cagg_ss = self.aggregate_consumption(self.C_ss)
        self.cbar_frac = jnp.array(cbar_frac, dtype=self.precision)
        self.c_bar = self.cbar_frac * self.Cagg_ss
        self._apply_subsistence_cap = float(cbar_frac) > 0
        self.reward_ss = self.period_utility(self.Cagg_ss)
        self.value_ss = self.reward_ss / (1 - self.beta)
        self.initial_state_mode = "box"
        self._stationary_chol = None

    def _set_state_scale(self, state_sd):
        self.state_sd = jnp.asarray(state_sd, dtype=self.precision)
        self.obs_sd = self.state_sd

    def set_scales(self, state_sd, policies_sd):
        """Set the state and policy scale vectors used for normalization."""
        self._set_state_scale(state_sd)
        self.policies_sd = jnp.asarray(policies_sd, dtype=self.precision)
        self.policy_sd = self.policies_sd

    def set_stationary_start(self, cov_obs):
        """Draw training initials from N(0, cov) in current observation coordinates."""
        cov = np.asarray(cov_obs, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-12 * np.eye(cov.shape[0])
        chol = np.linalg.cholesky(cov + jitter)
        self._stationary_chol = jnp.asarray(chol, dtype=self.precision)
        self.initial_state_mode = "stationary"
        self._check_stationary_feasibility()

    def _check_stationary_feasibility(self):
        if not self._apply_subsistence_cap:
            return
        if self._stationary_chol is None:
            return
        cov = self._stationary_chol @ self._stationary_chol.T
        sd = jnp.sqrt(jnp.maximum(jnp.diag(cov), 0))
        obs = -4.0 * sd
        capital, productivity = self._capital_and_productivity(obs)
        output = self.production(capital, productivity)
        c_bar = jnp.asarray(self.min_feasible_output(), dtype=self.precision)
        if bool(jnp.any(output <= c_bar)):
            raise ValueError(
                "a 4-sd stationary draw has output at or below subsistence "
                f"(output={np.asarray(output)}, c_bar={np.asarray(c_bar)})"
            )

    @abstractmethod
    def deterministic_steady_state_action(self):
        """Return the action that implements steady-state investment."""

    @abstractmethod
    def allocation_from_policy(
        self, policy, output, project_investment=None, beta=None, hard_floor=None
    ):
        """Map a model-specific policy output into investment and consumption."""

    def multiplier_from_policy(
        self, policy, consumption=None, beta=None, hard_floor=None
    ):
        policy = jnp.asarray(policy, dtype=self.precision)
        return jnp.zeros(
            policy.shape[:-1] + (self.n_sectors,), dtype=self.precision
        )

    def naive_projected_investment(self, unconstrained_investment):
        return jnp.maximum(unconstrained_investment, self.I_min)

    def investment_shortfall(self, investment):
        return jnp.maximum(self.I_min - investment, 0)

    def investment_slack(self, investment):
        return (investment - self.I_min) / self.I_ss

    def constraint_binds(self, investment, tol=1e-3):
        slack = self.investment_slack(investment)
        return (self.i_min_frac > 0) & (
            slack <= jnp.array(tol, dtype=self.precision)
        )

    def aggregate_consumption(self, consumption):
        inv_sigma = self.sigma_c ** (-1)
        return jnp.sum(
            self.xi**inv_sigma * consumption ** (1 - inv_sigma), axis=-1
        ) ** (1 / (1 - inv_sigma))

    def _capital_and_productivity(self, state):
        state_notnorm = state * self.state_sd + self.state_ss
        capital = jnp.exp(state_notnorm[..., : self.n_sectors])
        productivity = state_notnorm[..., self.n_sectors :]
        return capital, productivity

    def _normalize_state(self, capital, productivity):
        state_notnorm = jnp.concatenate(
            [jnp.log(capital), productivity], axis=-1
        )
        return (state_notnorm - self.state_ss) / self.state_sd

    def production(self, capital, productivity):
        return jnp.exp(productivity) * capital**self.alpha

    def next_capital(self, capital, investment):
        return (
            (1 - self.delta) * capital
            + investment
            - (self.phi / 2)
            * (investment / capital - self.delta) ** 2
            * capital
        )

    def _subsistence_consumption(self, consumption):
        tiny = jnp.array(1e-8, dtype=self.precision)
        return jnp.maximum(consumption - self.c_bar, tiny)

    def period_utility(self, aggregate_consumption):
        sigma = self.risk_aversion
        excess = self._subsistence_consumption(aggregate_consumption)
        return excess ** (1 - sigma) / (1 - sigma)

    def marginal_utility(self, consumption):
        return self._subsistence_consumption(consumption) ** (-self.risk_aversion)

    def subsistence_saving_rate_cap(self, output):
        output = jnp.maximum(output, jnp.array(1e-12, dtype=self.precision))
        slack = jnp.array(1e-6, dtype=self.precision)
        cap = 1 - self.c_bar / output - slack
        return jnp.clip(cap, self.saving_rate_min + slack, self.saving_rate_max)

    def reward(
        self,
        state,
        policy,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        _, consumption, _ = self.allocation_from_policy(
            policy, output, project_investment, beta, hard_floor
        )
        return self.period_utility(self.aggregate_consumption(consumption))

    def discounted_period_weight(self, horizon):
        return jnp.sum(self.discount_rate ** jnp.arange(horizon))

    def deterministic_steady_state_welfare(self, horizon):
        return self.reward_ss * self.discounted_period_weight(horizon)

    def consumption_equivalent(self, welfare, baseline_welfare, horizon):
        sigma = self.risk_aversion
        weight = self.discounted_period_weight(horizon)
        exponent = 1 / (1 - sigma)

        def level(w):
            return self.c_bar + ((1 - sigma) * w / weight) ** exponent

        return level(welfare) / level(baseline_welfare) - 1

    def min_feasible_output(self):
        return self.c_bar

    def initial_state(self, rng, init_range=0, init_range_a=None, mode=None):
        resolved = self.initial_state_mode if mode is None else mode
        if resolved == "stationary":
            if self._stationary_chol is None:
                raise ValueError("stationary start requires set_stationary_start(cov_obs)")
            noise = random.normal(rng, shape=(self.dim_states,), dtype=self.precision)
            return self._stationary_chol @ noise
        rng_k, rng_a = random.split(rng, 2)
        if init_range_a is None:
            init_range_a = init_range
        range_k = init_range / 100
        range_a = init_range_a / 100
        if init_range > 0:
            capital = random.uniform(
                rng_k,
                shape=(self.n_sectors,),
                minval=(1 - range_k) * self.K_ss,
                maxval=(1 + range_k) * self.K_ss,
                dtype=self.precision,
            )
        else:
            capital = self.K_ss
        if init_range_a > 0:
            productivity_level = random.uniform(
                rng_a,
                shape=(self.n_sectors,),
                minval=(1 - range_a) * self.A_ss,
                maxval=(1 + range_a) * self.A_ss,
                dtype=self.precision,
            )
        else:
            productivity_level = self.A_ss
        return self._normalize_state(capital, jnp.log(productivity_level))

    def step(
        self,
        state,
        policy,
        shock,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        investment, _, _ = self.allocation_from_policy(
            policy, output, project_investment, beta, hard_floor
        )
        productivity_next = self.rho * productivity + self.shock_sd * shock
        capital_next = self.next_capital(capital, investment)
        return self._normalize_state(capital_next, productivity_next)

    def expect_realization(
        self, state_next, policy_next, beta=None, hard_floor=None
    ):
        capital, productivity = self._capital_and_productivity(state_next)
        output = self.production(capital, productivity)
        investment, consumption, _ = self.allocation_from_policy(
            policy_next, output, None, beta, hard_floor
        )
        utility_prime = self.marginal_utility(consumption)
        capital_price = utility_prime / (
            1 - self.phi * (investment / capital - self.delta)
        )
        multiplier = (
            self.multiplier_from_policy(
                policy_next, consumption, beta, hard_floor
            )
            * utility_prime
        )
        return (
            utility_prime
            * (
                self.alpha
                * jnp.exp(productivity)
                * capital ** (self.alpha - 1)
            )
            + capital_price
            * (
                (1 - self.delta)
                + (self.phi / 2)
                * (investment**2 / capital**2 - self.delta**2)
            )
            - (1 - self.delta) * multiplier
        )

    def capital_price(self, state, policy, beta=None, hard_floor=None):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        investment, consumption, _ = self.allocation_from_policy(
            policy, output, None, beta, hard_floor
        )
        return self.marginal_utility(consumption) / (
            1 - self.phi * (investment / capital - self.delta)
        )

    def unconstrained_euler_residual(
        self, state, policy, expect, beta=None, hard_floor=None
    ):
        return (
            self.capital_price(state, policy, beta, hard_floor)
            / (self.beta * expect)
            - 1
        )

    def euler_residual(
        self, state, policy, expect, beta=None, hard_floor=None
    ):
        return self.unconstrained_euler_residual(
            state, policy, expect, beta, hard_floor
        )

    def static_normalized_euler_residual(
        self, consumption, multiplier, capital, output
    ):
        utility_prime = self.marginal_utility(consumption)
        rhs = (
            self.beta
            * utility_prime
            * (self.alpha * output / capital + (1 - self.delta))
        )
        rhs = rhs - self.beta * (1 - self.delta) * multiplier
        return (utility_prime - multiplier - rhs) / utility_prime

    def terminal_value(self, state, horizon=512):
        """Differentiable fixed-saving tail under conditional-mean shocks."""
        capital, productivity = self._capital_and_productivity(state)
        return self.ss_saving_tail_value(
            capital, productivity, n_periods=horizon
        )

    def ss_saving_tail_value(self, capital, productivity, n_periods=512):
        def period(carry, _):
            current_capital, current_productivity, discount, value = carry
            output = self.production(
                current_capital, current_productivity
            )
            investment = self.s_ss * output
            consumption = output - investment
            reward = self.period_utility(
                self.aggregate_consumption(consumption)
            )
            return (
                self.next_capital(current_capital, investment),
                self.rho * current_productivity,
                discount * self.beta,
                value + discount * reward,
            ), None

        init = (
            jnp.asarray(capital, dtype=self.precision),
            jnp.asarray(productivity, dtype=self.precision),
            jnp.ones((), dtype=self.precision),
            jnp.zeros((), dtype=self.precision),
        )
        (_, _, _, value), _ = lax.scan(
            period, init, None, length=int(n_periods)
        )
        return value

    def ss_saving_tail_from_state(self, state, n_periods=512):
        return self.terminal_value(state, horizon=n_periods)

    def option_value(
        self,
        state,
        policy,
        expected_next_mu,
        beta=None,
        hard_floor=None,
    ):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        _, consumption, _ = self.allocation_from_policy(
            policy, output, None, beta, hard_floor
        )
        return (
            self.beta
            * (1 - self.delta)
            * expected_next_mu
            / self.marginal_utility(consumption)
        )

    def euler_error(self, state, policy, expect):
        residual = jnp.reshape(
            self.euler_residual(state, policy, expect), (-1,)
        )
        mean_loss = jnp.mean(residual**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(residual))
        min_accuracy = jnp.min(1 - jnp.abs(residual))
        accuracies = 1 - jnp.abs(residual)
        return (
            mean_loss,
            mean_accuracy,
            min_accuracy,
            accuracies,
            accuracies,
        )

    def monte_carlo_expect(
        self, state, policy, next_policy_fn, rng, mc_draws
    ):
        shocks = self.mc_shocks(rng, mc_draws)
        next_states = vmap(
            lambda shock: self.step(state, policy, shock)
        )(shocks)
        next_policies = vmap(next_policy_fn)(next_states)
        return jnp.mean(
            vmap(self.expect_realization)(next_states, next_policies), axis=0
        )

    def loss(self, state, expect, policy):
        return self.euler_error(state, policy, expect)

    def sample_shock(self, rng, n_draws=1):
        if n_draws == 1:
            return random.normal(
                rng, shape=(self.n_sectors,), dtype=self.precision
            )
        return random.normal(
            rng, shape=(n_draws, self.n_sectors), dtype=self.precision
        )

    def mc_shocks(self, rng=None, mc_draws=8):
        if rng is None:
            rng = random.PRNGKey(0)
        if mc_draws % 2 == 0:
            base = random.normal(
                rng,
                shape=(mc_draws // 2, self.n_sectors),
                dtype=self.precision,
            )
            return jnp.concatenate([base, -base], axis=0)
        return random.normal(
            rng, shape=(mc_draws, self.n_sectors), dtype=self.precision
        )

    def get_aggregates(self, simul_policies, simul_states):
        policies = jnp.atleast_2d(simul_policies)
        states = jnp.atleast_2d(simul_states)
        capital, productivity = self._capital_and_productivity(states)
        output = self.production(capital, productivity)
        investment, consumption, _ = self.allocation_from_policy(
            policies, output
        )
        return {
            "C": jnp.log(
                jnp.mean(consumption, axis=-1) / jnp.mean(self.C_ss)
            ),
            "K": jnp.log(
                jnp.mean(capital, axis=-1) / jnp.mean(self.K_ss)
            ),
            "I": jnp.log(
                jnp.mean(investment, axis=-1) / jnp.mean(self.I_ss)
            ),
            "Y": jnp.log(
                jnp.mean(output, axis=-1) / jnp.mean(self.Y_ss)
            ),
            "A": jnp.log(jnp.mean(jnp.exp(productivity), axis=-1)),
        }
