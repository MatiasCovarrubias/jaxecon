from jax import numpy as jnp

from DEQN.econ_models.RBC.exact_kink_model import ExactKinkModel
from DEQN.econ_models.RBC.model import Model
from DEQN.econ_models.RBC.projected_irreversible_model import (
    ProjectedIrreversibleModel,
)

from .base import WelfareEnvironment


class RbcMultiSector(WelfareEnvironment):
    """APG environment wrapper around the shared RBC model.

    The economics live in ``DEQN.econ_models.RBC``; this class picks the model
    variant from the constructor arguments and exposes the rollout interface
    required by ``WelfareEnvironment``. Model attributes (``beta``, ``K_ss``,
    ``discount_rate``, ``value_ss``, ``obs_ss``, ``action_dim``, ...) are
    delegated to ``self.econ`` and can be read directly from the environment.

    Model selection:

    - ``policy_map="exact_kink"``: ``ExactKinkModel`` (explicit irreversibility kink).
    - ``i_min_frac > 0``: ``ProjectedIrreversibleModel`` (investment floor by projection).
    - otherwise: ``Model`` (smooth saving-rate policy).
    """

    def __init__(
        self,
        N=1,
        investment_penalty=0.0,
        project_investment=True,
        policy_map="saving_rate",
        policy_beta=10.0,
        hard_floor=False,
        kappa_mu=None,
        state_k_scale=0.1,
        investment_cap_frac=0.9,
        **kwargs,
    ):
        if investment_penalty < 0:
            raise ValueError("investment_penalty must be nonnegative")
        if investment_penalty > 0 and project_investment:
            raise ValueError("a positive investment penalty requires project_investment=False")
        if investment_penalty > 0 and N != 1:
            raise ValueError("the utility-scaled investment penalty currently supports one sector")
        if policy_map not in ("saving_rate", "exact_kink"):
            raise ValueError(
                "policy_map must be 'saving_rate' or 'exact_kink'"
            )
        self.investment_penalty = float(investment_penalty)
        if policy_map == "exact_kink":
            self.econ = ExactKinkModel(
                n_sectors=N,
                project_investment=project_investment,
                policy_beta=policy_beta,
                hard_floor=hard_floor,
                kappa_mu=kappa_mu,
                state_k_scale=state_k_scale,
                investment_cap_frac=investment_cap_frac,
                **kwargs,
            )
        elif float(kwargs.get("i_min_frac", 0.0)) > 0:
            self.econ = ProjectedIrreversibleModel(
                n_sectors=N,
                project_investment=project_investment,
                **kwargs,
            )
        else:
            self.econ = Model(
                n_sectors=N,
                project_investment=project_investment,
                **kwargs,
            )
        self.N = self.econ.n_sectors

    def __getattr__(self, name):
        if name == "econ":
            raise AttributeError(name)
        return getattr(self.econ, name)

    def deterministic_steady_state_action(self):
        return self.econ.deterministic_steady_state_action()

    def terminal_value(self, state, horizon=512):
        return self.econ.terminal_value(state, horizon)

    def saving_rate_from_action(self, action, output=None):
        return self.econ.saving_rate_from_policy(action, output)

    def allocation_from_action(self, action, output):
        return self.econ.allocation_from_policy(action, output)

    def training_reward(self, state, action):
        reward = self.econ.reward(state, action)
        if not self.investment_penalty:
            return reward
        K, a = self.econ._capital_and_productivity(state)
        output = self.econ.production(K, a)
        investment, _, _ = self.econ.allocation_from_policy(action, output)
        shortfall = self.econ.investment_shortfall(investment)
        utility_scale = self.econ.marginal_utility(self.econ.C_ss)
        return reward - self.investment_penalty * jnp.sum(utility_scale * shortfall, axis=-1)

    def normalized_investment_slack(self, state, action):
        K, a = self.econ._capital_and_productivity(state)
        output = self.econ.production(K, a)
        investment, _, _ = self.econ.allocation_from_policy(action, output)
        return (investment - self.econ.I_min) / self.econ.I_ss

    def constraint_utility_scale(self):
        return self.econ.marginal_utility(self.econ.C_ss) * self.econ.I_ss

    def constraint_state_grid(
        self,
        size=33,
        k_min=0.90,
        k_max=1.16,
        a_sd_min=-2.5,
        a_sd_max=2.5,
    ):
        if size < 2:
            raise ValueError("constraint grid size must be at least two")
        k = self.econ.K_ss[0] * jnp.linspace(k_min, k_max, size, dtype=self.econ.precision)
        stationary_a_sd = self.econ.shock_sd[0] / jnp.sqrt(1 - self.econ.rho[0] ** 2)
        a = stationary_a_sd * jnp.linspace(
            a_sd_min,
            a_sd_max,
            size,
            dtype=self.econ.precision,
        )
        capital, productivity = jnp.meshgrid(k, a, indexing="ij")
        return self.econ._normalize_state(
            capital.reshape(-1, 1),
            productivity.reshape(-1, 1),
        )

    def deterministic_steady_state_welfare(self, horizon):
        return self.econ.deterministic_steady_state_welfare(horizon)

    def consumption_equivalent(self, welfare, baseline_welfare, horizon):
        return self.econ.consumption_equivalent(welfare, baseline_welfare, horizon)

    def initial_state(self, rng, init_range=0, init_range_a=None, mode=None):
        return self.econ.initial_state(
            rng, init_range, init_range_a=init_range_a, mode=mode
        )

    def set_stationary_start(self, cov_obs):
        return self.econ.set_stationary_start(cov_obs)

    def sample_shock(self, rng):
        return self.econ.sample_shock(rng)

    def transition(self, state, action, shock):
        return self.econ.step(state, action, shock)

    def reset(self, rng, init_range=5, init_range_a=None):
        obs_init = self.initial_state(rng, init_range=init_range, init_range_a=init_range_a)
        return obs_init, obs_init

    def step(self, rng, state, action):
        shock = self.sample_shock(rng)
        reward = self.training_reward(state, action)
        new_obs = self.transition(state, action, shock)
        done = jnp.array(False)
        info = jnp.array([0.0])
        return new_obs, new_obs, reward, done, info
