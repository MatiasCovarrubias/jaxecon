"""DEQN RBC model with an investment floor and KKT multiplier policy."""

from jax import nn
from jax import numpy as jnp

from DEQN.econ_models.RBC.projected_irreversible_model import (
    ProjectedIrreversibleModel,
)


class IrreversibleModel(ProjectedIrreversibleModel):
    """Saving-rate and normalized-multiplier policy for irreversible DEQN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.irreversible:
            raise ValueError(
                "IrreversibleModel requires a positive i_min_frac"
            )
        self.dim_policies = 2 * self.n_sectors
        self.deqn_policy_dim = self.dim_policies
        self.action_dim = self.n_sectors
        self.apg_action_dim = self.action_dim

    def multiplier_from_policy(
        self, policy, consumption=None, beta=None, hard_floor=None
    ):
        del consumption, beta, hard_floor
        policy = jnp.asarray(policy, dtype=self.precision)
        if policy.shape[-1] <= self.n_sectors:
            return jnp.zeros(
                policy.shape[:-1] + (self.n_sectors,),
                dtype=self.precision,
            )
        return nn.softplus(policy[..., self.n_sectors :])

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
        complementarity = jnp.minimum(
            self.investment_slack(investment), mu_n
        )
        return jnp.concatenate(
            [jnp.reshape(foc, (-1,)), jnp.reshape(complementarity, (-1,))]
        )
