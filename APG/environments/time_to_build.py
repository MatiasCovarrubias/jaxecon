"""Kydland and Prescott (1982) time-to-build economy as an APG environment.

Planner problem of Kydland and Prescott, "Time to Build and Aggregate
Fluctuations", Econometrica 50(6), under full information. Default parameters
are their Table I on a quarterly calendar.

Technology. ``J`` periods are needed to build productive capital. ``s_j`` is
the stock of projects ``j`` periods from completion (``j = 1, ..., J``);
``s_J`` are the projects started this period. Stage ``j`` absorbs the fraction
``phi_j`` of a project's value, so the spending on projects already under way,
``sum_{j<J} phi_j s_j``, is committed before the period starts::

    k'    = (1 - delta) k + s_1
    s_j'  = s_{j+1}                                   j = 1, ..., J - 1
    Y     = exp(z) n^theta k^(1-theta) [(1 - sigma) + sigma (k / m)^nu]^(-(1-theta)/nu)
    c + sum_j phi_j s_j + m' - m = Y

with ``m`` inventories (a factor of production), ``n`` hours, and
``z = z_p + z_q`` log productivity with a persistent and a transitory part,
``z_p' = rho z_p + shock_sd e_1`` and ``z_q' = transitory_shock_sd e_2``.
Kydland and Prescott write the shock as additive around one; the log form
agrees to first order and keeps productivity positive.

Preferences. ``u(c, L) = (c^mu L^(1-mu))^gamma / gamma`` with the leisure
aggregate ``L = 1 - alpha0 n - (1 - alpha0) a`` and ``a' = (1 - eta) a + eta n``.
``a`` is the distributed lag of past hours that makes utility non-separable
over time. ``gamma = 0`` is the logarithmic case.

Actions. Every action passes through a bounded sigmoid whose zero maps to the
deterministic steady state, so the allocation satisfies the resource
constraint and stays strictly positive by construction:

* the share ``w`` of free resources ``R = Y + m - sum_{j<J} phi_j s_j`` carried
  over as inventories, ``m' = w R``, when ``inventories`` is on;
* the saving rate ``s`` out of the remainder: new starts absorb
  ``phi_J s_J = s (1 - w) R`` and consumption is ``c = (1 - s)(1 - w) R``;
* hours ``n`` in ``(hours_min, hours_max)`` when ``variable_labor`` is on.

With inventories off, ``s`` is the share of new-project spending in
consumption plus new-project spending, the analogue of the RBC saving rate.

``R`` is positive whenever committed spending is below output plus
inventories. That holds in the neighbourhood of the steady state (committed
spending is about 18% of output) and under bounded saving rates it fails only
after output collapses by a large factor. ``R`` is floored at
``free_resources_floor * R_ss`` as a numerical guard; ``free_resources_slack``
reports how far a state is from that floor.

Information. All period-``t`` decisions observe ``z_t``. Kydland and Prescott's
noisy productivity indicator and the two-stage timing within the period (hours
and starts before the shock is revealed, consumption after) are not modelled.

State. Logs of ``k, s_1, ..., s_{J-1}, m, a`` and the levels ``z_p, z_q``,
as deviations from steady state divided by ``state_sd``. Switching a feature
off removes its state and action coordinates.
"""

import math

from jax import lax, nn, random
from jax import numpy as jnp

from .base import WelfareEnvironment


class TimeToBuildRbc(WelfareEnvironment):
    """Kydland and Prescott's economy with policy maps that are feasible by design."""

    def __init__(
        self,
        n_stages=4,
        stage_shares=None,
        beta=0.99,
        delta=0.025,
        theta=0.64,
        gamma=-0.5,
        consumption_weight=1 / 3,
        alpha0=0.5,
        eta=0.10,
        variable_labor=True,
        hours_ss=None,
        inventories=True,
        sigma=0.28e-5,
        nu=4.0,
        rho=0.95,
        shock_sd=0.00902,
        transitory_shock_sd=0.00182,
        volatility_scale=1.0,
        saving_rate_min=1e-6,
        saving_rate_max=1 - 1e-6,
        inventory_share_min=1e-6,
        inventory_share_max=1 - 1e-6,
        hours_min=1e-3,
        hours_max=1 - 1e-3,
        free_resources_floor=1e-3,
        precision=jnp.float32,
        double_precision=False,
        discount_rate=None,
    ):
        J = int(n_stages)
        if J < 1:
            raise ValueError("n_stages must be a positive integer")
        shares = [1.0 / J] * J if stage_shares is None else [float(x) for x in stage_shares]
        if len(shares) != J or min(shares) <= 0 or abs(sum(shares) - 1.0) > 1e-10:
            raise ValueError("stage_shares must have n_stages positive entries summing to one")
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not 0 < delta <= 1:
            raise ValueError("delta must be in (0, 1]")
        if not 0 < theta < 1:
            raise ValueError("theta must be in (0, 1)")
        if gamma >= 1:
            raise ValueError("gamma must be below one")
        if not 0 < consumption_weight < 1:
            raise ValueError("consumption_weight must be in (0, 1)")
        if not 0 < alpha0 <= 1:
            raise ValueError("alpha0 must be in (0, 1]")
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        if inventories and not (0 < sigma < 1 and nu > 0):
            raise ValueError("inventories need 0 < sigma < 1 and nu > 0")
        if not 0 < saving_rate_min < saving_rate_max < 1:
            raise ValueError("saving-rate bounds must satisfy 0 < min < max < 1")
        if not 0 < inventory_share_min < inventory_share_max < 1:
            raise ValueError("inventory-share bounds must satisfy 0 < min < max < 1")
        if not 0 < hours_min < hours_max < 1:
            raise ValueError("hours bounds must satisfy 0 < min < max < 1")
        if not 0 < free_resources_floor < 1:
            raise ValueError("free_resources_floor must be in (0, 1)")
        if shock_sd < 0 or transitory_shock_sd < 0:
            raise ValueError("shock standard deviations must be nonnegative")

        self.precision = jnp.float64 if double_precision else precision
        self.n_stages = J
        self.variable_labor = bool(variable_labor)
        self.inventories = bool(inventories)
        self.has_transitory = float(transitory_shock_sd) > 0
        self.n_shocks = 2 if self.has_transitory else 1

        # Steady state in Python floats (Kydland and Prescott, Section 4).
        r = 1.0 / beta - 1.0
        q = sum(shares[j] * (1.0 + r) ** j for j in range(J))
        user_cost = q * (r + delta)
        sigma_eff = sigma if self.inventories else 0.0
        if self.inventories:
            b1 = (sigma * user_cost / ((1.0 - sigma) * r)) ** (1.0 / (1.0 + nu))
            b2 = (1.0 - sigma) + sigma * b1 ** (-nu)
            b2_power = b2 ** (-(1.0 - theta) / nu)
        else:
            b1, b2, b2_power = 0.0, 1.0, 1.0
        b3 = ((1.0 - theta) * (1.0 - sigma_eff) * b2_power / b2 / user_cost) ** (1.0 / theta)
        b4 = b3 ** (1.0 - theta) * b2_power
        consumption_share = 1.0 - delta * b3 / b4
        leisure_weight = alpha0 + (1.0 - alpha0) * eta / (r + eta)
        mu = consumption_weight
        if hours_ss is None:
            hours_ss = theta / (theta + (1.0 - mu) / mu * leisure_weight * consumption_share)
        if not 0 < hours_ss < 1:
            raise ValueError("steady-state hours must be in (0, 1)")
        if not hours_min < hours_ss < hours_max:
            raise ValueError("steady-state hours must lie strictly inside the hours bounds")

        K_ss = b3 * hours_ss
        Y_ss = b4 * hours_ss
        Inv_ss = b1 * K_ss
        Starts_ss = delta * K_ss
        C_ss = Y_ss - Starts_ss
        committed_ss = sum(shares[j] for j in range(J - 1)) * Starts_ss
        R_ss = Y_ss + Inv_ss - committed_ss
        new_spend_ss = shares[J - 1] * Starts_ss
        omega_ss = Inv_ss / R_ss if self.inventories else 0.0
        s_ss = new_spend_ss / (R_ss - Inv_ss)
        if not saving_rate_min < s_ss < saving_rate_max:
            raise ValueError("steady-state saving rate must lie strictly inside the saving-rate bounds")
        if self.inventories and not inventory_share_min < omega_ss < inventory_share_max:
            raise ValueError("steady-state inventory share must lie strictly inside its bounds")

        f = lambda x: jnp.array(x, dtype=self.precision)
        self.beta = f(beta)
        self.delta = f(delta)
        self.theta = f(theta)
        self.gamma = f(gamma)
        self.gamma_is_zero = float(gamma) == 0.0
        self.mu = f(mu)
        self.alpha0 = f(alpha0)
        self.eta = f(eta)
        self.sigma = f(sigma_eff)
        self.nu = f(nu)
        self.rho = f(rho)
        self.shock_sd = f(shock_sd * volatility_scale)
        self.transitory_shock_sd = f(transitory_shock_sd * volatility_scale)
        self.stage_shares = jnp.array(shares, dtype=self.precision)
        self.phi_new = self.stage_shares[J - 1]
        self.committed_shares = self.stage_shares[: J - 1]
        self.saving_rate_min = f(saving_rate_min)
        self.saving_rate_max = f(saving_rate_max)
        self.inventory_share_min = f(inventory_share_min)
        self.inventory_share_max = f(inventory_share_max)
        self.hours_min = f(hours_min)
        self.hours_max = f(hours_max)
        self.discount_rate = self.beta if discount_rate is None else f(discount_rate)

        self.r_ss = r
        self.q_ss = q
        self.K_ss = f(K_ss)
        self.Y_ss = f(Y_ss)
        self.C_ss = f(C_ss)
        self.I_ss = f(Starts_ss)
        self.Starts_ss = f(Starts_ss)
        self.Inv_ss = f(Inv_ss)
        self.N_ss = f(hours_ss)
        self.L_ss = f(1.0 - hours_ss)
        self.R_ss = f(R_ss)
        self.s_ss = f(s_ss)
        self.omega_ss = f(omega_ss)
        self.free_resources_floor = f(free_resources_floor * R_ss)
        self.steady_state = {
            "capital": K_ss,
            "output": Y_ss,
            "consumption": C_ss,
            "investment": Starts_ss,
            "starts": Starts_ss,
            "inventories": Inv_ss,
            "hours": hours_ss,
            "free_resources": R_ss,
            "saving_rate": s_ss,
            "inventory_share": omega_ss,
            "interest_rate": r,
            "capital_price": q,
            "capital_output_ratio": K_ss / Y_ss,
        }

        # State layout.
        layout = [("capital", 1)]
        if J > 1:
            layout.append(("projects", J - 1))
        if self.inventories:
            layout.append(("inventories", 1))
        if self.variable_labor:
            layout.append(("leisure_stock", 1))
        layout.append(("productivity", 1))
        if self.has_transitory:
            layout.append(("transitory", 1))
        self._slices = {}
        start = 0
        for name, size in layout:
            self._slices[name] = slice(start, start + size)
            start += size
        self.state_names = [name for name, _ in layout]
        self.dim_states = start
        self.state_dim = start
        self.obs_dim = start

        ss_parts = [jnp.log(self.K_ss)[None]]
        if J > 1:
            ss_parts.append(jnp.full((J - 1,), math.log(Starts_ss), dtype=self.precision))
        if self.inventories:
            ss_parts.append(jnp.log(self.Inv_ss)[None])
        if self.variable_labor:
            ss_parts.append(jnp.log(self.N_ss)[None])
        ss_parts.append(jnp.zeros((1,), dtype=self.precision))
        if self.has_transitory:
            ss_parts.append(jnp.zeros((1,), dtype=self.precision))
        self.state_ss = jnp.concatenate(ss_parts)
        self.state_sd = jnp.ones(self.dim_states, dtype=self.precision)
        self.obs_ss = self.state_ss
        self.obs_sd = self.state_sd

        # Action layout: logit deviations; zero reproduces the steady state.
        self.action_names = ["saving_rate"]
        logits = [_logit((s_ss - saving_rate_min) / (saving_rate_max - saving_rate_min))]
        if self.variable_labor:
            self.action_names.append("hours")
            logits.append(_logit((hours_ss - hours_min) / (hours_max - hours_min)))
        if self.inventories:
            self.action_names.append("inventory_share")
            logits.append(
                _logit((omega_ss - inventory_share_min) / (inventory_share_max - inventory_share_min))
            )
        self._action_index = {name: i for i, name in enumerate(self.action_names)}
        self.action_dim = len(self.action_names)
        self.dim_policies = self.action_dim
        self.policies_ss = jnp.array(logits, dtype=self.precision)
        self.policies_sd = jnp.ones(self.action_dim, dtype=self.precision)
        self.policy_ss = self.policies_ss
        self.policy_sd = self.policies_sd

        self.reward_ss = self.period_utility(self.C_ss, self.L_ss)
        self.value_ss = self.reward_ss / (1 - self.beta)

    # ------------------------------------------------------------------ scales

    def set_scales(self, state_sd, policies_sd):
        """Set the state and action scales used to normalize network inputs and outputs."""
        self.state_sd = jnp.asarray(state_sd, dtype=self.precision)
        self.obs_sd = self.state_sd
        self.policies_sd = jnp.asarray(policies_sd, dtype=self.precision)
        self.policy_sd = self.policies_sd

    # ------------------------------------------------------------------ state

    def levels(self, state):
        """Unnormalize a state into a dict of levels (logs for the productivity components)."""
        raw = jnp.asarray(state, dtype=self.precision) * self.state_sd + self.state_ss
        out = {}
        for name, sl in self._slices.items():
            x = raw[..., sl]
            out[name] = x if name in ("productivity", "transitory") else jnp.exp(x)
        if "projects" not in out:
            out["projects"] = jnp.zeros(raw.shape[:-1] + (0,), dtype=self.precision)
        if "inventories" not in out:
            out["inventories"] = jnp.zeros_like(out["capital"])
        if "leisure_stock" not in out:
            out["leisure_stock"] = jnp.broadcast_to(self.N_ss, out["capital"].shape)
        if "transitory" not in out:
            out["transitory"] = jnp.zeros_like(out["productivity"])
        return out

    def normalize(self, levels):
        parts = [jnp.log(levels["capital"])]
        if self.n_stages > 1:
            parts.append(jnp.log(levels["projects"]))
        if self.inventories:
            parts.append(jnp.log(levels["inventories"]))
        if self.variable_labor:
            parts.append(jnp.log(levels["leisure_stock"]))
        parts.append(levels["productivity"])
        if self.has_transitory:
            parts.append(levels["transitory"])
        raw = jnp.concatenate(parts, axis=-1)
        return (raw - self.state_ss) / self.state_sd

    # ------------------------------------------------------------------ actions

    def _action(self, action, name):
        action = jnp.asarray(action, dtype=self.precision)
        i = self._action_index[name]
        return action[..., i : i + 1] * self.policies_sd[i] + self.policies_ss[i]

    def saving_rate_from_action(self, action):
        return self.saving_rate_min + (self.saving_rate_max - self.saving_rate_min) * nn.sigmoid(
            self._action(action, "saving_rate")
        )

    def hours_from_action(self, action):
        if not self.variable_labor:
            shape = jnp.asarray(action).shape[:-1] + (1,)
            return jnp.broadcast_to(self.N_ss, shape)
        return self.hours_min + (self.hours_max - self.hours_min) * nn.sigmoid(
            self._action(action, "hours")
        )

    def inventory_share_from_action(self, action):
        if not self.inventories:
            shape = jnp.asarray(action).shape[:-1] + (1,)
            return jnp.zeros(shape, dtype=self.precision)
        return self.inventory_share_min + (
            self.inventory_share_max - self.inventory_share_min
        ) * nn.sigmoid(self._action(action, "inventory_share"))

    def deterministic_steady_state_action(self):
        return jnp.zeros(self.action_dim, dtype=self.precision)

    # ------------------------------------------------------------------ economics

    def production(self, capital, inventories, hours, log_productivity):
        base = jnp.exp(log_productivity) * hours**self.theta * capital ** (1 - self.theta)
        if not self.inventories:
            return base
        mix = (1 - self.sigma) + self.sigma * (capital / inventories) ** self.nu
        return base * mix ** (-(1 - self.theta) / self.nu)

    def committed_spending(self, projects):
        return jnp.sum(self.committed_shares * projects, axis=-1, keepdims=True)

    def leisure(self, hours, leisure_stock):
        return 1 - self.alpha0 * hours - (1 - self.alpha0) * leisure_stock

    def period_utility(self, consumption, leisure):
        if self.gamma_is_zero:
            return self.mu * jnp.log(consumption) + (1 - self.mu) * jnp.log(leisure)
        composite = consumption**self.mu * leisure ** (1 - self.mu)
        return composite**self.gamma / self.gamma

    def allocation(self, state, action):
        """All period flows implied by a state and an action.

        Returns a dict with ``output, committed, free_resources_raw,
        free_resources, inventory_share, inventories_next, saving_rate,
        consumption, new_spending, starts, investment, hours, leisure``
        (all shaped ``(..., 1)``).
        """
        lv = self.levels(state)
        hours = self.hours_from_action(action)
        log_productivity = lv["productivity"] + lv["transitory"]
        output = self.production(lv["capital"], lv["inventories"], hours, log_productivity)
        committed = self.committed_spending(lv["projects"])
        free_raw = output + lv["inventories"] - committed
        free = jnp.maximum(free_raw, self.free_resources_floor)
        inventory_share = self.inventory_share_from_action(action)
        inventories_next = inventory_share * free
        remaining = (1 - inventory_share) * free
        saving_rate = self.saving_rate_from_action(action)
        new_spending = saving_rate * remaining
        consumption = (1 - saving_rate) * remaining
        starts = new_spending / self.phi_new
        investment = committed + new_spending + inventories_next - lv["inventories"]
        return {
            "output": output,
            "committed": committed,
            "free_resources_raw": free_raw,
            "free_resources": free,
            "inventory_share": inventory_share,
            "inventories_next": inventories_next,
            "saving_rate": saving_rate,
            "consumption": consumption,
            "new_spending": new_spending,
            "starts": starts,
            "investment": investment,
            "hours": hours,
            "leisure": self.leisure(hours, lv["leisure_stock"]),
        }

    def training_reward(self, state, action):
        flows = self.allocation(state, action)
        return self.period_utility(flows["consumption"], flows["leisure"])[..., 0]

    def transition(self, state, action, shock):
        lv = self.levels(state)
        flows = self.allocation(state, action)
        shock = jnp.asarray(shock, dtype=self.precision)
        first_stage = lv["projects"][..., :1] if self.n_stages > 1 else flows["starts"]
        capital_next = (1 - self.delta) * lv["capital"] + first_stage
        projects_next = jnp.concatenate([lv["projects"][..., 1:], flows["starts"]], axis=-1)
        leisure_stock_next = (1 - self.eta) * lv["leisure_stock"] + self.eta * flows["hours"]
        productivity_next = self.rho * lv["productivity"] + self.shock_sd * shock[..., :1]
        transitory_next = (
            self.transitory_shock_sd * shock[..., 1:2]
            if self.has_transitory
            else jnp.zeros_like(productivity_next)
        )
        return self.normalize(
            {
                "capital": capital_next,
                "projects": projects_next,
                "inventories": flows["inventories_next"],
                "leisure_stock": leisure_stock_next,
                "productivity": productivity_next,
                "transitory": transitory_next,
            }
        )

    def free_resources_slack(self, state, action):
        """Distance of free resources from the floor, relative to steady-state free resources."""
        flows = self.allocation(state, action)
        return (flows["free_resources_raw"] - self.free_resources_floor)[..., 0] / self.R_ss

    # ------------------------------------------------------------------ shocks and starts

    def sample_shock(self, rng, n_draws=1):
        shape = (self.n_shocks,) if n_draws == 1 else (n_draws, self.n_shocks)
        return random.normal(rng, shape=shape, dtype=self.precision)

    def initial_state(self, rng, init_range=0, mode=None):
        """Draw levels within ``init_range`` percent of steady state; zero returns the steady state."""
        keys = random.split(rng, 6)
        frac = init_range / 100

        def draw(key, center, size=1):
            center = jnp.broadcast_to(jnp.asarray(center, dtype=self.precision), (size,))
            if init_range <= 0:
                return center
            return random.uniform(
                key,
                shape=(size,),
                minval=(1 - frac) * center,
                maxval=(1 + frac) * center,
                dtype=self.precision,
            )

        one = jnp.ones((), dtype=self.precision)
        return self.normalize(
            {
                "capital": draw(keys[0], self.K_ss),
                "projects": draw(keys[1], self.Starts_ss, max(self.n_stages - 1, 0)),
                "inventories": draw(keys[2], self.Inv_ss if self.inventories else one),
                "leisure_stock": draw(keys[3], self.N_ss),
                "productivity": jnp.log(draw(keys[4], one)),
                "transitory": jnp.log(draw(keys[5], one)),
            }
        )

    # ------------------------------------------------------------------ welfare

    def discounted_period_weight(self, horizon):
        return jnp.sum(self.discount_rate ** jnp.arange(horizon))

    def deterministic_steady_state_welfare(self, horizon):
        return self.reward_ss * self.discounted_period_weight(horizon)

    def consumption_equivalent(self, welfare, baseline_welfare, horizon):
        """Fractional consumption change to the baseline path that yields ``welfare``."""
        if self.gamma_is_zero:
            return jnp.exp((welfare - baseline_welfare) / (self.mu * self.discounted_period_weight(horizon))) - 1
        return (welfare / baseline_welfare) ** (1 / (self.mu * self.gamma)) - 1

    def terminal_value(self, state, horizon=512):
        """Discounted utility of steady-state actions under conditional-mean shocks."""
        action = self.deterministic_steady_state_action()
        shock = jnp.zeros((self.n_shocks,), dtype=self.precision)

        def period(carry, _):
            obs, discount, value = carry
            value = value + discount * self.training_reward(obs, action)
            return (self.transition(obs, action, shock), discount * self.beta, value), None

        init = (
            jnp.asarray(state, dtype=self.precision),
            jnp.ones((), dtype=self.precision),
            jnp.zeros((), dtype=self.precision),
        )
        (_, _, value), _ = lax.scan(period, init, None, length=int(horizon))
        return value

    # ------------------------------------------------------------------ diagnostics

    def get_aggregates(self, simul_policies, simul_states):
        """Log deviations from steady state of the main aggregates along a simulation."""
        states = jnp.atleast_2d(simul_states)
        actions = jnp.atleast_2d(simul_policies)
        lv = self.levels(states)
        flows = self.allocation(states, actions)
        out = {
            "C": jnp.log(flows["consumption"][..., 0] / self.C_ss),
            "K": jnp.log(lv["capital"][..., 0] / self.K_ss),
            "Y": jnp.log(flows["output"][..., 0] / self.Y_ss),
            "I": jnp.log(flows["investment"][..., 0] / self.Starts_ss),
            "S": jnp.log(flows["starts"][..., 0] / self.Starts_ss),
            "N": jnp.log(flows["hours"][..., 0] / self.N_ss),
            "A": (lv["productivity"] + lv["transitory"])[..., 0],
        }
        out["Y/N"] = out["Y"] - out["N"]
        if self.inventories:
            out["M"] = jnp.log(lv["inventories"][..., 0] / self.Inv_ss)
        return out

    # ------------------------------------------------------------------ gym-style helpers

    def reset(self, rng, init_range=5):
        obs = self.initial_state(rng, init_range=init_range)
        return obs, obs

    def step(self, rng, state, action):
        shock = self.sample_shock(rng)
        reward = self.training_reward(state, action)
        new_obs = self.transition(state, action, shock)
        return new_obs, new_obs, reward, jnp.array(False), jnp.array([0.0])


def _logit(p):
    if not 0 < p < 1:
        raise ValueError("steady-state share must lie strictly inside its bounds")
    return math.log(p) - math.log1p(-p)
