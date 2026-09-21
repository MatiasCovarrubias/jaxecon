"""Kydland-Prescott time-to-build environment: contract, steady state, accounting, timing, nesting."""

import unittest

import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import random

from APG.environments import RbcMultiSector, TimeToBuildRbc, check_environment
from APG.loglinear.solve import linearize_environment, solve_loglinear

jax_config.update("jax_enable_x64", True)

VARIANTS = {
    "kydland_prescott": dict(),
    "no_inventories": dict(inventories=False),
    "fixed_labor": dict(variable_labor=False),
    "three_stages": dict(n_stages=3),
    "unequal_stages": dict(n_stages=3, stage_shares=[0.5, 0.3, 0.2]),
    "log_utility": dict(gamma=0.0),
    "minimal": dict(n_stages=1, inventories=False, variable_labor=False, transitory_shock_sd=0.0),
}


def _env(**kwargs):
    values = dict(double_precision=True)
    values.update(kwargs)
    return TimeToBuildRbc(**values)


class ContractTest(unittest.TestCase):
    def test_every_variant_satisfies_the_contract(self):
        for name, kwargs in VARIANTS.items():
            with self.subTest(variant=name):
                check_environment(_env(**kwargs))

    def test_single_precision_default_satisfies_the_contract(self):
        check_environment(TimeToBuildRbc())

    def test_state_and_action_layout_follow_the_switches(self):
        env = _env()
        self.assertEqual(
            env.state_names,
            ["capital", "projects", "inventories", "leisure_stock", "productivity", "transitory"],
        )
        self.assertEqual(env.action_names, ["saving_rate", "hours", "inventory_share"])
        self.assertEqual(env.dim_states, 1 + 3 + 1 + 1 + 1 + 1)
        minimal = _env(**VARIANTS["minimal"])
        self.assertEqual(minimal.state_names, ["capital", "productivity"])
        self.assertEqual(minimal.action_names, ["saving_rate"])

    def test_invalid_parameters_are_rejected(self):
        for kwargs in (
            dict(n_stages=0),
            dict(n_stages=2, stage_shares=[0.7, 0.2]),
            dict(hours_min=0.5, hours_max=0.4),
            dict(saving_rate_min=0.0),
            dict(gamma=1.0),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TimeToBuildRbc(**kwargs)


class SteadyStateTest(unittest.TestCase):
    def test_kydland_prescott_calibration_ratios(self):
        ss = _env().steady_state
        self.assertAlmostEqual(ss["capital_output_ratio"] / 4, 2.46, places=2)
        self.assertAlmostEqual(ss["inventories"] / ss["capital"], 0.10, places=2)
        self.assertAlmostEqual(ss["hours"], 0.308, places=3)
        self.assertAlmostEqual(ss["investment"] / ss["output"], 0.246, places=3)
        self.assertAlmostEqual(ss["capital_price"], 1.0153, places=4)
        self.assertAlmostEqual(ss["interest_rate"], 1 / 0.99 - 1, places=12)

    def test_planner_first_order_conditions_hold_at_the_steady_state(self):
        for name, kwargs in VARIANTS.items():
            with self.subTest(variant=name):
                lin = linearize_environment(_env(**kwargs))
                np.testing.assert_allclose(lin["foc_residual"], 0.0, atol=1e-10)

    def test_zero_action_and_zero_shock_hold_the_steady_state_for_many_periods(self):
        env = _env()
        state = env.initial_state(random.PRNGKey(0), 0)
        action = env.deterministic_steady_state_action()
        shock = jnp.zeros(env.n_shocks)
        for _ in range(50):
            state = env.transition(state, action, shock)
        np.testing.assert_allclose(state, 0.0, atol=1e-10)

    def test_lq_solution_is_stable_and_concave(self):
        sol = solve_loglinear(_env())
        self.assertLess(sol.spectral_radius, 1.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(sol.R) < 0))
        self.assertEqual(sol.C.shape, (3, 8))


class AllocationTest(unittest.TestCase):
    def test_resource_constraint_holds_with_equality(self):
        env = _env()
        rng = random.PRNGKey(1)
        for i in range(5):
            k_state, k_action = random.split(random.fold_in(rng, i))
            state = env.initial_state(k_state, 20)
            action = 0.5 * random.normal(k_action, (env.action_dim,))
            lv = env.levels(state)
            flows = env.allocation(state, action)
            spending = (
                flows["consumption"]
                + jnp.sum(env.stage_shares[:-1] * lv["projects"], axis=-1, keepdims=True)
                + env.phi_new * flows["starts"]
                + flows["inventories_next"]
                - lv["inventories"]
            )
            np.testing.assert_allclose(spending, flows["output"], rtol=1e-12)
            np.testing.assert_allclose(
                flows["investment"], flows["output"] - flows["consumption"], rtol=1e-12
            )

    def test_extreme_actions_keep_every_flow_positive(self):
        for kwargs in (dict(), dict(inventories=False), dict(variable_labor=False)):
            env = TimeToBuildRbc(**kwargs)
            state = env.initial_state(random.PRNGKey(0), 0)
            for value in (-1e6, 1e6):
                action = jnp.full((env.action_dim,), value)
                flows = env.allocation(state, action)
                for key in ("consumption", "starts", "leisure", "hours"):
                    self.assertTrue(bool(jnp.all(flows[key] > 0)), key)
                if env.inventories:
                    self.assertTrue(bool(jnp.all(flows["inventories_next"] > 0)))
                self.assertTrue(bool(jnp.all(flows["hours"] < 1)))
                self.assertTrue(bool(jnp.isfinite(env.training_reward(state, action))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(env.transition(state, action, jnp.zeros(env.n_shocks))))))

    def test_free_resources_floor_is_far_from_the_steady_state(self):
        env = _env()
        state = env.initial_state(random.PRNGKey(0), 0)
        slack = env.free_resources_slack(state, env.deterministic_steady_state_action())
        self.assertGreater(float(slack), 0.9)

    def test_consumption_equivalent_inverts_a_consumption_scaling(self):
        for kwargs in (dict(), dict(gamma=0.0)):
            env = _env(**kwargs)
            horizon = 32
            weight = float(env.discounted_period_weight(horizon))
            baseline = float(env.reward_ss) * weight
            scaled = float(env.period_utility(1.02 * env.C_ss, env.L_ss)) * weight
            np.testing.assert_allclose(env.consumption_equivalent(scaled, baseline, horizon), 0.02, rtol=1e-10)


class TimingTest(unittest.TestCase):
    def test_new_projects_reach_capital_after_exactly_n_stages(self):
        for J in (1, 3, 4):
            with self.subTest(n_stages=J):
                env = _env(n_stages=J, inventories=False, variable_labor=False, transitory_shock_sd=0.0)
                zero_action = env.deterministic_steady_state_action()
                shock = jnp.zeros(env.n_shocks)
                state = env.initial_state(random.PRNGKey(0), 0)
                impulse = zero_action.at[0].set(0.5)
                path = []
                state = env.transition(state, impulse, shock)
                path.append(float(env.levels(state)["capital"][0]))
                for _ in range(J + 1):
                    state = env.transition(state, zero_action, shock)
                    path.append(float(env.levels(state)["capital"][0]))
                K_ss = float(env.K_ss)
                for t in range(J - 1):
                    self.assertAlmostEqual(path[t], K_ss, places=10)
                self.assertGreater(path[J - 1], K_ss * (1 + 1e-4))

    def test_committed_spending_is_predetermined(self):
        env = _env()
        state = env.initial_state(random.PRNGKey(0), 10)
        base = env.allocation(state, env.deterministic_steady_state_action())["committed"]
        other = env.allocation(state, jnp.array([2.0, -1.0, 0.5]))["committed"]
        np.testing.assert_allclose(base, other)


class NestingTest(unittest.TestCase):
    def test_one_stage_fixed_labor_no_inventories_is_the_standard_rbc(self):
        gamma, mu = -0.5, 1 / 3
        ttb = _env(
            n_stages=1,
            inventories=False,
            variable_labor=False,
            transitory_shock_sd=0.0,
            gamma=gamma,
            consumption_weight=mu,
        )
        rbc = RbcMultiSector(
            N=1,
            beta=0.99,
            alpha=1 - 0.64,
            delta=0.025,
            phi=0.0,
            eps_c=1 / (1 - mu * gamma),
            rho=0.95,
            shock_sd=0.00902,
            double_precision=True,
            precision=jnp.float64,
        )
        np.testing.assert_allclose(float(ttb.K_ss / ttb.Y_ss), float(rbc.K_ss[0] / rbc.Y_ss[0]), rtol=1e-12)
        np.testing.assert_allclose(float(ttb.s_ss), float(rbc.s_ss[0]), rtol=1e-12)
        sol_ttb = solve_loglinear(ttb)
        sol_rbc = solve_loglinear(rbc)
        np.testing.assert_allclose(sol_ttb.C, sol_rbc.C, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(sol_ttb.states_sd, sol_rbc.states_sd, rtol=1e-8)
        np.testing.assert_allclose(sol_ttb.policies_sd, sol_rbc.policies_sd, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
