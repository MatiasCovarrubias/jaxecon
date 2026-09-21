"""Economic invariants of the shared RBC models in DEQN.econ_models.RBC."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from DEQN.econ_models.RBC.euler_eval import create_euler_eval_fn
from DEQN.econ_models.RBC.irreversible_model import IrreversibleModel
from DEQN.econ_models.RBC.model import Model
from DEQN.econ_models.RBC.projected_irreversible_model import ProjectedIrreversibleModel
from DEQN.econ_models.RBC.welfare_eval import create_welfare_eval_fn


def _zero_policy(model):
    return lambda params, obs: jnp.zeros(model.dim_policies, dtype=model.precision)


class SteadyStateTest(unittest.TestCase):
    def test_zero_policy_is_a_fixed_point_without_shocks(self):
        model = Model(n_sectors=1, shock_sd=0.0)
        state = jnp.zeros(model.dim_states)
        next_state = model.step(state, jnp.zeros(model.dim_policies), jnp.zeros(model.n_sectors))
        np.testing.assert_allclose(next_state, state, rtol=1e-5, atol=1e-6)

    def test_preferences_are_ces_and_reward_ss_matches_closed_form(self):
        model = Model(n_sectors=1)
        np.testing.assert_allclose(model.risk_aversion, 1 / model.eps_c)
        c_ss = jnp.mean(model.Cagg_ss)
        sigma = float(model.risk_aversion)
        np.testing.assert_allclose(model.reward_ss, c_ss ** (1 - sigma) / (1 - sigma), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.marginal_utility(model.C_ss), model.C_ss ** (-sigma), rtol=1e-5, atol=1e-6)

    def test_consumption_equivalent_matches_ces_homogeneity(self):
        model = Model(n_sectors=1)
        lam = 0.01
        weight = model.discounted_period_weight(32)
        welfare = ((1 + lam) ** (1 - model.risk_aversion)) * model.reward_ss * weight
        baseline = model.reward_ss * weight
        np.testing.assert_allclose(model.consumption_equivalent(welfare, baseline, 32), lam, rtol=1e-5, atol=1e-6)

    def test_terminal_value_is_differentiable_and_matches_tail(self):
        model = Model(n_sectors=1, shock_sd=0.02)
        state = jnp.array([0.02, -0.01])
        value = model.terminal_value(state, horizon=16)
        gradient = jax.grad(lambda s: model.terminal_value(s, horizon=16))(state)
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.0)
        np.testing.assert_allclose(value, model.ss_saving_tail_from_state(state, n_periods=16))

    def test_ss_saving_tail_at_steady_state_is_finite_horizon_welfare(self):
        model = Model(n_sectors=1, shock_sd=0.0, phi=0.0)
        tail = model.ss_saving_tail_value(model.K_ss, model.a_ss, n_periods=64)
        np.testing.assert_allclose(tail, model.deterministic_steady_state_welfare(64), rtol=1e-5, atol=1e-5)


class EulerTest(unittest.TestCase):
    def test_residual_is_zero_at_steady_state(self):
        model = Model(n_sectors=1, shock_sd=0.0)
        state = jnp.zeros(model.dim_states)
        policy = jnp.zeros(model.dim_policies)
        expect = model.expect_realization(state, policy)
        np.testing.assert_allclose(model.euler_residual(state, policy, expect), 0.0, atol=1e-5)
        mse, acc, *_ = model.euler_error(state, policy, expect)
        np.testing.assert_allclose(mse, 0.0, atol=1e-10)
        np.testing.assert_allclose(acc, 1.0, atol=1e-5)

    def test_default_model_uses_unconstrained_euler(self):
        model = Model(n_sectors=1, shock_sd=0.0)
        state = jnp.zeros(model.dim_states)
        policy = jnp.zeros(model.dim_policies)
        expect = model.expect_realization(state, policy)
        np.testing.assert_allclose(
            model.euler_residual(state, policy, expect), model.unconstrained_euler_residual(state, policy, expect)
        )
        np.testing.assert_allclose(model.I_min, 0.0)

    def test_zero_policy_euler_eval_is_exact_without_shocks(self):
        model = Model(n_sectors=1, shock_sd=0.0)
        eval_fn = create_euler_eval_fn(
            model, policy_fn=_zero_policy(model), periods_per_epis=8, n_epis=4, mc_draws=8, init_range=0
        )
        mse, acc, min_acc = eval_fn({}, random.PRNGKey(0))
        np.testing.assert_allclose(mse, 0.0, atol=1e-8)
        np.testing.assert_allclose(acc, 1.0, atol=1e-5)
        np.testing.assert_allclose(min_acc, 1.0, atol=1e-5)

    def test_mc_shocks_are_antithetic_when_even(self):
        model = Model(n_sectors=2, shock_sd=0.02)
        shocks = model.mc_shocks(random.PRNGKey(0), mc_draws=8)
        self.assertEqual(tuple(shocks.shape), (8, 2))
        np.testing.assert_allclose(shocks[4:], -shocks[:4], atol=0)


class WelfareEvalTest(unittest.TestCase):
    def test_zero_policy_without_shocks_reproduces_steady_state(self):
        model = Model(n_sectors=1, shock_sd=0.0)
        eval_fn = create_welfare_eval_fn(model, policy_fn=_zero_policy(model), horizon=32, n_epis=8, init_range=0)
        m = eval_fn({}, random.PRNGKey(0))
        np.testing.assert_allclose(m.welfare, m.welfare_ss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m.ce_vs_no_shocks_ss, 0.0, atol=1e-6)
        np.testing.assert_allclose(m.ce_vs_fixed_saving_rate, 0.0, atol=1e-6)
        np.testing.assert_allclose(m.K_rel, 1.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m.s_mean, model.s_ss, rtol=1e-5, atol=1e-6)
        for vol in (m.std_c, m.std_y, m.std_i):
            np.testing.assert_allclose(vol, 0.0, atol=1e-6)

    def test_zero_policy_with_shocks_is_the_fixed_saving_rate_baseline(self):
        model = Model(n_sectors=1, shock_sd=0.02)
        eval_fn = create_welfare_eval_fn(model, policy_fn=_zero_policy(model), horizon=64, n_epis=32, init_range=0)
        m = eval_fn({}, random.PRNGKey(1))
        np.testing.assert_allclose(m.welfare, m.welfare_fixed_s, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m.ce_vs_fixed_saving_rate, 0.0, atol=1e-6)
        np.testing.assert_allclose(m.std_c, m.std_y, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(m.std_i, m.std_y, rtol=1e-4, atol=1e-5)
        self.assertGreater(float(m.std_y), 0.0)

    def test_unprojected_policy_reports_violations_and_baseline_is_projected(self):
        model = ProjectedIrreversibleModel(n_sectors=1, shock_sd=0.0, phi=0.0, i_min_frac=0.975, project_investment=False)
        eval_fn = create_welfare_eval_fn(
            model, policy_fn=lambda params, obs: jnp.full(model.action_dim, -20.0), horizon=4, n_epis=2, init_range=0
        )
        m = eval_fn({}, random.PRNGKey(4))
        self.assertGreater(float(m.i_violation_frac), 0.0)
        self.assertGreater(float(m.i_violation_mean), 0.0)
        np.testing.assert_allclose(m.welfare_fixed_s, m.welfare_ss, rtol=1e-5, atol=1e-6)


class IrreversibilityTest(unittest.TestCase):
    def test_positive_floor_requires_an_irreversible_model_class(self):
        with self.assertRaisesRegex(ValueError, "ProjectedIrreversibleModel"):
            Model(n_sectors=1, i_min_frac=0.8)

    def test_projection_clips_and_unprojected_exposes_shortfall(self):
        projected = ProjectedIrreversibleModel(n_sectors=1, i_min_frac=0.8, project_investment=True)
        unprojected = ProjectedIrreversibleModel(n_sectors=1, i_min_frac=0.8, project_investment=False)
        action = jnp.full((1,), -20.0)
        clipped = projected.allocation_from_policy(action, projected.Y_ss)[0]
        raw = unprojected.allocation_from_policy(action, unprojected.Y_ss)[0]
        self.assertTrue(bool(jnp.all(clipped >= projected.I_min)))
        self.assertTrue(bool(jnp.all(raw < unprojected.I_min)))
        self.assertGreater(float(jnp.squeeze(unprojected.investment_shortfall(raw))), 0.0)

    def test_zero_policy_at_steady_state_stays_interior(self):
        model = IrreversibleModel(n_sectors=1, shock_sd=0.0, phi=0.0, i_min_frac=0.975)
        policy = jnp.zeros(model.dim_policies)
        investment, _, saving_rate = model.allocation_from_policy(policy, model.Y_ss)
        np.testing.assert_allclose(saving_rate, model.s_ss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(investment, model.I_ss, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(jnp.squeeze(investment)), float(jnp.squeeze(model.I_min)))
        next_state = model.step(jnp.zeros(model.dim_states), policy, jnp.zeros(model.n_sectors))
        np.testing.assert_allclose(next_state, jnp.zeros(model.dim_states), rtol=1e-5, atol=1e-6)

    def test_low_output_projects_investment_onto_the_floor(self):
        model = IrreversibleModel(n_sectors=1, phi=0.0, i_min_frac=0.975)
        output = 0.5 * model.Y_ss
        policy = jnp.zeros(model.dim_policies)
        unconstrained = model.unconstrained_saving_rate(policy) * output
        investment, consumption, _ = model.allocation_from_policy(policy, output)
        self.assertLess(float(jnp.squeeze(unconstrained)), float(jnp.squeeze(model.I_min)))
        np.testing.assert_allclose(investment, model.I_min, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool(jnp.all(consumption > 0)))
        self.assertTrue(bool(jnp.all(model.constraint_binds(investment))))

    def test_kkt_residual_is_zero_at_steady_state(self):
        model = IrreversibleModel(n_sectors=1, shock_sd=0.0, phi=0.0, i_min_frac=0.975)
        state = jnp.zeros(model.dim_states)
        policy = jnp.zeros(model.action_dim)
        expect = model.expect_realization(state, policy)
        np.testing.assert_allclose(model.unconstrained_euler_residual(state, policy, expect), 0.0, atol=1e-5)
        np.testing.assert_allclose(model.euler_residual(state, policy, expect), 0.0, atol=1e-5)
        np.testing.assert_allclose(model.multiplier_from_policy(policy), 0.0, atol=1e-6)

    def test_kkt_residual_needs_a_positive_multiplier_on_the_floor(self):
        model = IrreversibleModel(n_sectors=1, phi=0.0, i_min_frac=0.975)
        K = 1.15 * model.K_ss
        a = jnp.full_like(model.a_ss, -0.08)
        state = model._normalize_state(K, a)
        saving_logit = jnp.array([-20.0], dtype=model.precision)
        output = model.production(K, a)
        investment, _, _ = model.allocation_from_policy(saving_logit, output)
        next_state = model.step(state, saving_logit, jnp.zeros(model.n_sectors))
        expect = model.expect_realization(next_state, saving_logit)
        unconstrained = model.unconstrained_euler_residual(state, saving_logit, expect)

        np.testing.assert_allclose(investment, jnp.minimum(model.I_min, output * model.saving_rate_max), rtol=1e-4)
        self.assertGreater(float(jnp.squeeze(unconstrained)), 0.0)
        self.assertGreater(float(jnp.squeeze(model.euler_residual(state, saving_logit, expect)[0])), 0.0)

        mu_n = unconstrained / (unconstrained + 1)
        mu_logit = jnp.log(jnp.expm1(mu_n))
        kkt = model.euler_residual(state, jnp.concatenate([saving_logit, mu_logit]), expect)
        np.testing.assert_allclose(kkt, 0.0, atol=1e-4)


class StoneGearyTest(unittest.TestCase):
    def test_zero_subsistence_matches_crra_formulas(self):
        model = Model(n_sectors=1, cbar_frac=0.0)
        sigma = float(model.risk_aversion)
        c = model.Cagg_ss
        policy = jnp.zeros(model.dim_policies)
        old_u = c ** (1 - sigma) / (1 - sigma)
        old_mu = model.C_ss ** (-sigma)
        weight = model.discounted_period_weight(32)
        welfare = (1.01 ** (1 - sigma)) * model.reward_ss * weight
        baseline = model.reward_ss * weight
        old_ce = (welfare / baseline) ** (1 / (1 - sigma)) - 1
        np.testing.assert_allclose(model.period_utility(c), old_u, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(model.marginal_utility(model.C_ss), old_mu, rtol=1e-6, atol=1e-8)
        for horizon in (8, 32, 128):
            np.testing.assert_allclose(
                model.consumption_equivalent(welfare, baseline, horizon),
                old_ce,
                rtol=1e-5,
                atol=1e-6,
            )
        np.testing.assert_allclose(
            model.saving_rate_from_policy(policy, model.Y_ss),
            model.unconstrained_saving_rate(policy),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_subsistence_leaves_steady_state_and_changes_utility(self):
        baseline = Model(n_sectors=1, cbar_frac=0.0)
        model = Model(n_sectors=1, cbar_frac=0.6)
        sigma = float(model.risk_aversion)
        excess = model.Cagg_ss - model.c_bar
        np.testing.assert_allclose(model.K_ss, baseline.K_ss, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(model.s_ss, baseline.s_ss, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(model.c_bar, 0.6 * model.Cagg_ss, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            model.period_utility(model.Cagg_ss),
            excess ** (1 - sigma) / (1 - sigma),
            rtol=1e-6,
            atol=1e-8,
        )
        weight = model.discounted_period_weight(64)
        baseline_w = model.reward_ss * weight
        np.testing.assert_allclose(
            model.consumption_equivalent(baseline_w, baseline_w, 64), 0.0, atol=1e-6
        )
        worse = 1.1 * baseline_w
        better = 0.9 * baseline_w
        self.assertLess(
            float(model.consumption_equivalent(worse, baseline_w, 64)),
            float(model.consumption_equivalent(better, baseline_w, 64)),
        )
        capital = 0.3 * model.K_ss
        output = model.production(capital, model.a_ss)
        greedy = jnp.full(model.dim_policies, 20.0, dtype=model.precision)
        _, consumption, _ = model.allocation_from_policy(greedy, output)
        self.assertTrue(bool(jnp.all(consumption > model.c_bar)))

    def test_zero_saving_cannot_clear_subsistence_when_output_is_too_small(self):
        model = Model(n_sectors=1, cbar_frac=0.4)
        output = 0.5 * model.c_bar
        starve = jnp.full(model.dim_policies, -20.0, dtype=model.precision)
        _, consumption, saving_rate = model.allocation_from_policy(starve, output)
        self.assertTrue(bool(jnp.all(consumption <= output)))
        self.assertTrue(bool(jnp.all(consumption < model.c_bar)))
        self.assertTrue(bool(jnp.all(saving_rate >= model.saving_rate_min)))

    def test_cbar_frac_must_be_in_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "cbar_frac"):
            Model(n_sectors=1, cbar_frac=1.0)
        with self.assertRaisesRegex(ValueError, "cbar_frac"):
            Model(n_sectors=1, cbar_frac=-0.1)


if __name__ == "__main__":
    unittest.main()
