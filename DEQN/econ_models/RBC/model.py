"""Standard smooth saving-rate RBC model."""

import math
from typing import NamedTuple

import jax
from jax import nn
from jax import numpy as jnp

from DEQN.econ_models.RBC.base_model import RbcModelBase


class ActionEulerMapping(NamedTuple):
    """Economic derivative mapping from saving logits to next capital."""

    investment: jax.Array
    d_investment_d_action: jax.Array
    d_next_capital_d_investment: jax.Array
    M: jax.Array
    cap_binding: jax.Array


def softplus_beta(x, beta):
    """Numerically stable beta-scaled softplus."""
    beta = jnp.asarray(beta, dtype=x.dtype)
    scaled = beta * x
    denominator = jnp.maximum(
        beta, jnp.array(1e-8, dtype=x.dtype)
    )
    return jnp.where(
        scaled > 20, x, nn.softplus(scaled) / denominator
    )


def inverse_softplus_beta(y, beta):
    """Inverse of softplus_beta for positive inputs."""
    beta = jnp.asarray(beta, dtype=y.dtype)
    scaled = beta * y
    denominator = jnp.maximum(
        beta, jnp.array(1e-8, dtype=y.dtype)
    )
    return jnp.where(
        scaled > 20, y, jnp.log(jnp.expm1(scaled)) / denominator
    )


def excess_from_latent(latent, beta, hard_floor=False):
    if hard_floor:
        return jnp.maximum(
            latent, jnp.array(0, dtype=latent.dtype)
        )
    return softplus_beta(latent, beta)


def unit_excess(latent, beta, hard_floor=False):
    raw = excess_from_latent(latent, beta, hard_floor)
    if hard_floor:
        return raw
    unit = softplus_beta(
        jnp.ones((), dtype=latent.dtype),
        jnp.asarray(beta, dtype=latent.dtype),
    )
    return raw / jnp.maximum(
        unit, jnp.array(1e-12, dtype=latent.dtype)
    )


class _SavingRateModel(RbcModelBase):
    """Shared saving-rate implementation for explicit RBC variants."""

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
        saving_rate_min=1e-6,
        saving_rate_max=1 - 1e-6,
        project_investment=True,
        volatility_scale=1.0,
        double_precision=False,
        discount_rate=None,
        cbar_frac=0.0,
    ):
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
        self.saving_rate_min = jnp.array(
            saving_rate_min, dtype=self.precision
        )
        self.saving_rate_max = jnp.array(
            saving_rate_max, dtype=self.precision
        )
        self.project_investment = bool(project_investment)

        scaled_s_ss = (
            (self.s_ss - self.saving_rate_min)
            / (self.saving_rate_max - self.saving_rate_min)
        )
        if not bool(jnp.all((scaled_s_ss > 0) & (scaled_s_ss < 1))):
            raise ValueError(
                "steady-state saving rate must lie strictly inside the configured bounds"
            )
        self.policies_ss = jnp.log(scaled_s_ss) - jnp.log1p(-scaled_s_ss)
        self.policies_sd = jnp.ones(
            self.n_sectors, dtype=self.precision
        )
        self.policy_ss = self.policies_ss
        self.policy_sd = self.policies_sd
        self.dim_policies = self.n_sectors
        self.action_dim = self.n_sectors
        self.apg_action_dim = self.action_dim
        self.deqn_policy_dim = self.dim_policies
        self.policy_map = "saving_rate"
        self.policy_beta = None
        self.hard_floor = False
        self.kappa_I = (self.I_ss - self.I_min) / self.I_ss
        self.kappa_mu = jnp.array(
            0.020 / math.log1p(math.e), dtype=self.precision
        )

    def deterministic_steady_state_action(self):
        return jnp.zeros(self.action_dim, dtype=self.precision)

    def _saving_policy(self, policy):
        policy = jnp.asarray(policy, dtype=self.precision)
        return policy[..., : self.n_sectors]

    def unconstrained_saving_rate(self, policy):
        logits = (
            self._saving_policy(policy) * self.policies_sd
            + self.policies_ss
        )
        return self.saving_rate_min + (
            self.saving_rate_max - self.saving_rate_min
        ) * nn.sigmoid(logits)

    def investment_floor_saving_rate(self, output):
        output = jnp.maximum(
            output, jnp.array(1e-12, dtype=self.precision)
        )
        floor = self.I_min / output
        cap = self.saving_rate_max - jnp.array(
            1e-6, dtype=self.precision
        )
        return jnp.clip(floor, self.saving_rate_min, cap)

    def saving_rate_from_policy(
        self,
        policy,
        output=None,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        del beta, hard_floor
        saving_rate = self.unconstrained_saving_rate(policy)
        project = (
            self.project_investment
            if project_investment is None
            else bool(project_investment)
        )
        if output is not None and project:
            saving_rate = jnp.maximum(
                saving_rate, self.investment_floor_saving_rate(output)
            )
        if output is not None and self._apply_subsistence_cap:
            saving_rate = jnp.minimum(
                saving_rate, self.subsistence_saving_rate_cap(output)
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
        saving_rate = self.saving_rate_from_policy(
            policy, output, project_investment, beta, hard_floor
        )
        investment = saving_rate * output
        consumption = (1 - saving_rate) * output
        return investment, consumption, saving_rate

    def action_euler_mapping(
        self,
        state,
        policy,
        project_investment=None,
        beta=None,
        hard_floor=None,
    ):
        """Map a normalized saving logit's derivative into capital units.

        On the smooth interior branch, ``M`` is
        ``(dI / dz) * (dK_next / dI)``. At an active investment floor or
        subsistence saving-rate cap, the local derivative and ``M`` are zero,
        and ``cap_binding`` reports that the Stage 1 interior theory does not
        apply.
        """
        del beta, hard_floor
        capital, productivity = self._capital_and_productivity(state)
        output = self.production(capital, productivity)
        policy = self._saving_policy(policy)
        logits = policy * self.policies_sd + self.policies_ss
        unit_rate = nn.sigmoid(logits)
        unconstrained_rate = self.saving_rate_min + (
            self.saving_rate_max - self.saving_rate_min
        ) * unit_rate

        project = (
            self.project_investment
            if project_investment is None
            else bool(project_investment)
        )
        floor_binding = jnp.zeros_like(unconstrained_rate, dtype=bool)
        rate_after_floor = unconstrained_rate
        if project:
            floor = self.investment_floor_saving_rate(output)
            floor_binding = unconstrained_rate <= floor
            rate_after_floor = jnp.maximum(unconstrained_rate, floor)

        subsistence_binding = jnp.zeros_like(unconstrained_rate, dtype=bool)
        saving_rate = rate_after_floor
        if self._apply_subsistence_cap:
            cap = self.subsistence_saving_rate_cap(output)
            subsistence_binding = rate_after_floor >= cap
            saving_rate = jnp.minimum(rate_after_floor, cap)

        cap_binding = floor_binding | subsistence_binding
        investment = saving_rate * output
        interior_d_investment = (
            output
            * (self.saving_rate_max - self.saving_rate_min)
            * unit_rate
            * (1 - unit_rate)
            * self.policies_sd
        )
        d_investment_d_action = jnp.where(
            cap_binding, jnp.zeros_like(interior_d_investment), interior_d_investment
        )
        d_next_capital_d_investment = 1 - self.phi * (
            investment / capital - self.delta
        )
        return ActionEulerMapping(
            investment=investment,
            d_investment_d_action=d_investment_d_action,
            d_next_capital_d_investment=d_next_capital_d_investment,
            M=d_investment_d_action * d_next_capital_d_investment,
            cap_binding=cap_binding,
        )


class Model(_SavingRateModel):
    """Standard smooth RBC without an investment floor."""

    def __init__(self, *args, i_min_frac=0.0, project_investment=True, **kwargs):
        if float(i_min_frac) != 0:
            raise ValueError(
                "Model is the standard RBC; use ProjectedIrreversibleModel "
                "or IrreversibleModel for a positive investment floor"
            )
        super().__init__(
            *args,
            i_min_frac=0.0,
            project_investment=project_investment,
            **kwargs,
        )
