"""RBC model with a latent exact-kink investment policy."""

import math

from jax import numpy as jnp

from DEQN.econ_models.RBC.base_model import RbcModelBase
from DEQN.econ_models.RBC.model import (
    excess_from_latent,
    inverse_softplus_beta,
    unit_excess,
)


class ExactKinkModel(RbcModelBase):
    """Latent switching parameterization for the APG exact-kink method."""

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
        policy_beta=10.0,
        hard_floor=False,
        kappa_mu=None,
        state_k_scale=0.1,
        investment_cap_frac=0.9,
        saving_rate_min=1e-6,
        saving_rate_max=1 - 1e-6,
        project_investment=True,
        cbar_frac=0.0,
    ):
        if float(policy_beta) <= 0:
            raise ValueError("policy_beta must be positive")
        if float(state_k_scale) <= 0:
            raise ValueError("state_k_scale must be positive")
        if not 0 < investment_cap_frac < 1:
            raise ValueError(
                "investment_cap_frac must satisfy 0 < cap < 1"
            )
        if not 0 < saving_rate_min < saving_rate_max < 1:
            raise ValueError(
                "saving-rate bounds must satisfy 0 < min < max < 1"
            )
        super().__init__(
            n_sectors=n_sectors,
            precision=precision,
            beta=beta,
            alpha=alpha,
            delta=delta,
            eps_c=eps_c,
            rho=rho,
            rho_values=rho_values,
            phi=phi,
            i_min_frac=i_min_frac,
            shock_sd=shock_sd,
            sigma_c=sigma_c,
            xi_values=xi_values,
            volatility_scale=volatility_scale,
            double_precision=double_precision,
            discount_rate=discount_rate,
            cbar_frac=cbar_frac,
        )
        self.policy_beta = jnp.array(
            policy_beta, dtype=self.precision
        )
        self.hard_floor = bool(hard_floor)
        self.state_k_scale = jnp.array(
            state_k_scale, dtype=self.precision
        )
        self.investment_cap_frac = jnp.array(
            investment_cap_frac, dtype=self.precision
        )
        self.saving_rate_min = jnp.array(
            saving_rate_min, dtype=self.precision
        )
        self.saving_rate_max = jnp.array(
            saving_rate_max, dtype=self.precision
        )
        self.project_investment = bool(project_investment)
        productivity_scale = jnp.maximum(
            self.shock_sd, jnp.array(1e-6, dtype=self.precision)
        )
        self._set_state_scale(
            jnp.concatenate(
                [
                    jnp.full(
                        self.n_sectors,
                        self.state_k_scale,
                        dtype=self.precision,
                    ),
                    productivity_scale,
                ]
            )
        )

        self.kappa_I = (self.I_ss - self.I_min) / self.I_ss
        softplus_one = jnp.array(
            math.log1p(math.e), dtype=self.precision
        )
        if kappa_mu is None:
            self.kappa_mu = jnp.array(
                0.020 / float(softplus_one), dtype=self.precision
            )
        else:
            if float(kappa_mu) <= 0:
                raise ValueError("kappa_mu must be positive")
            self.kappa_mu = jnp.array(
                kappa_mu, dtype=self.precision
            )

        scaled_s_ss = (
            (self.s_ss - self.saving_rate_min)
            / (self.saving_rate_max - self.saving_rate_min)
        )
        self.policies_ss = (
            jnp.log(scaled_s_ss) - jnp.log1p(-scaled_s_ss)
        )
        self.policies_sd = jnp.ones(
            self.n_sectors, dtype=self.precision
        )
        self.policy_ss = self.policies_ss
        self.policy_sd = self.policies_sd
        self.dim_policies = self.n_sectors
        self.action_dim = self.n_sectors
        self.apg_action_dim = self.action_dim
        self.deqn_policy_dim = self.dim_policies
        self.policy_map = "exact_kink"

    def deterministic_steady_state_action(self):
        return self.steady_state_latent()

    def _policy_latent(self, policy):
        policy = jnp.asarray(policy, dtype=self.precision)
        return policy[..., : self.n_sectors]

    def _excess_scale(self, beta=None, hard_floor=None):
        beta = (
            self.policy_beta
            if beta is None
            else jnp.asarray(beta, dtype=self.precision)
        )
        hard = (
            self.hard_floor
            if hard_floor is None
            else bool(hard_floor)
        )
        return beta, hard

    def steady_state_latent(self, beta=None, hard_floor=None):
        beta, hard = self._excess_scale(beta, hard_floor)
        target = (
            (self.I_ss - self.I_min)
            / (
                self.I_ss
                * jnp.maximum(
                    self.kappa_I,
                    jnp.array(1e-12, dtype=self.precision),
                )
            )
        )
        if hard:
            return target
        return inverse_softplus_beta(target, beta)

    def latent_regularization(self, policy, threshold=5.0):
        latent = self._policy_latent(policy)
        excess = jnp.maximum(
            jnp.abs(latent)
            - jnp.array(threshold, dtype=self.precision),
            0,
        )
        return jnp.mean(excess**2)

    def multiplier_from_latent(
        self, latent, beta=None, hard_floor=None
    ):
        beta, hard = self._excess_scale(beta, hard_floor)
        return (
            self.kappa_mu
            * self.marginal_utility(self.C_ss)
            * excess_from_latent(-latent, beta, hard)
        )

    def multiplier_from_policy(
        self, policy, consumption=None, beta=None, hard_floor=None
    ):
        multiplier = self.multiplier_from_latent(
            self._policy_latent(policy), beta, hard_floor
        )
        denominator = self.marginal_utility(
            self.C_ss if consumption is None else consumption
        )
        return multiplier / denominator

    def saving_rate_from_policy(
        self,
        policy,
        output=None,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        if output is None:
            raise ValueError(
                "exact-kink saving rates require current output"
            )
        _, _, saving_rate = self.allocation_from_policy(
            policy, output, project_investment, beta, hard_floor
        )
        return saving_rate

    def allocation_from_policy(
        self,
        policy,
        output,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        del project_investment
        beta, hard = self._excess_scale(beta, hard_floor)
        latent = self._policy_latent(policy)
        investment = (
            self.I_min
            + self.I_ss
            * self.kappa_I
            * unit_excess(latent, beta, hard)
        )
        investment = jnp.minimum(
            investment, self.investment_cap_frac * output
        )
        if self._apply_subsistence_cap:
            investment = jnp.minimum(
                investment, self.subsistence_saving_rate_cap(output) * output
            )
        consumption = output - investment
        saving_rate = investment / jnp.maximum(
            output, jnp.array(1e-12, dtype=self.precision)
        )
        return investment, consumption, saving_rate

    def safety_clip_active(
        self, policy, output, beta=None, hard_floor=None
    ):
        beta, hard = self._excess_scale(beta, hard_floor)
        uncapped = (
            self.I_min
            + self.I_ss
            * self.kappa_I
            * unit_excess(
                self._policy_latent(policy), beta, hard
            )
        )
        return uncapped > self.investment_cap_frac * output

    def normalized_euler_gap(
        self,
        state,
        policy,
        rhs,
        beta=None,
        hard_floor=None,
        mu=None,
    ):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        _, consumption, _ = self.allocation_from_policy(
            policy, output, None, beta, hard_floor
        )
        if mu is None:
            mu = self.multiplier_from_latent(
                self._policy_latent(policy), beta, hard_floor
            )
        utility_prime = self.marginal_utility(consumption)
        return (utility_prime - mu - rhs) / utility_prime

    def euler_residual(
        self, state, policy, expect, beta=None, hard_floor=None
    ):
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        investment, consumption, _ = self.allocation_from_policy(
            policy, output, None, beta, hard_floor
        )
        price = self.capital_price(state, policy, beta, hard_floor)
        mu_n = self.multiplier_from_policy(
            policy, consumption, beta, hard_floor
        )
        foc = price * (1 - mu_n) / (self.beta * expect) - 1
        if not self.irreversible:
            return foc
        complementarity = jnp.minimum(
            self.investment_slack(investment), mu_n
        )
        return jnp.concatenate(
            [jnp.reshape(foc, (-1,)), jnp.reshape(complementarity, (-1,))]
        )
