"""Economic invariants of ExactKinkModel: floor, complementarity, steady state, safety clip."""

import math
import unittest

import jax.numpy as jnp
import numpy as np

from DEQN.econ_models.RBC.exact_kink_model import ExactKinkModel
from DEQN.econ_models.RBC.model import inverse_softplus_beta, softplus_beta
from DEQN.econ_models.RBC.projected_irreversible_model import ProjectedIrreversibleModel


class SoftplusBetaTest(unittest.TestCase):
    def test_matches_relu_at_large_beta(self):
        x = jnp.array([-1.0, 1.5])
        np.testing.assert_allclose(softplus_beta(x, 500.0), jnp.maximum(x, 0.0), atol=1e-4)
        np.testing.assert_allclose(softplus_beta(jnp.array(0.0), 500.0), math.log(2) / 500.0, rtol=1e-5)

    def test_inverse_round_trip(self):
        y = jnp.array([0.2, 1.0, 1.3132617])
        np.testing.assert_allclose(softplus_beta(inverse_softplus_beta(y, 10.0), 10.0), y, rtol=1e-5, atol=1e-6)


class ExactKinkModelTest(unittest.TestCase):
    def test_investment_never_falls_below_the_floor(self):
        model = ExactKinkModel(n_sectors=1, phi=0.0, i_min_frac=0.8, policy_beta=20.0)
        for latent in (-8.0, -1.0, 0.0, 1.0, 4.0):
            investment, consumption, _ = model.allocation_from_policy(jnp.array([latent]), model.Y_ss)
            self.assertGreaterEqual(float(jnp.squeeze(investment)), float(jnp.squeeze(model.I_min)) - 1e-6)
            self.assertGreater(float(jnp.squeeze(consumption)), 0.0)
            np.testing.assert_allclose(investment + consumption, model.Y_ss, rtol=1e-6)

    def test_hard_floor_hits_the_bound_and_complementarity_holds(self):
        model = ExactKinkModel(n_sectors=1, phi=0.0, i_min_frac=0.8, hard_floor=True)
        investment, _, _ = model.allocation_from_policy(jnp.array([-2.0]), model.Y_ss)
        np.testing.assert_allclose(investment, model.I_min, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(jnp.squeeze(model.multiplier_from_latent(jnp.array([-2.0])))), 0.0)
        np.testing.assert_allclose(model.multiplier_from_latent(jnp.array([2.0])), 0.0, atol=1e-6)
        for latent in (-3.0, -0.5, 0.0, 0.8, 2.5):
            action = jnp.array([latent])
            investment, _, _ = model.allocation_from_policy(action, model.Y_ss)
            np.testing.assert_allclose((investment - model.I_min) * model.multiplier_from_latent(action), 0.0, atol=1e-6)

    def test_steady_state_action_reproduces_steady_state_for_any_beta(self):
        for beta in (10.0, 50.0, 500.0):
            model = ExactKinkModel(n_sectors=1, shock_sd=0.0, phi=0.0, i_min_frac=0.8, policy_beta=beta)
            action = model.deterministic_steady_state_action()
            np.testing.assert_allclose(action, 1.0, rtol=1e-5, atol=1e-6)
            investment, consumption, saving_rate = model.allocation_from_policy(action, model.Y_ss)
            np.testing.assert_allclose(investment, model.I_ss, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(consumption, model.C_ss, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(saving_rate, model.s_ss, rtol=1e-5, atol=1e-6)
            next_state = model.step(jnp.zeros(model.dim_states), action, jnp.zeros(model.n_sectors))
            np.testing.assert_allclose(next_state, jnp.zeros(model.dim_states), rtol=1e-5, atol=1e-6)

    def test_unit_latent_is_steady_state_without_a_floor(self):
        model = ExactKinkModel(n_sectors=1, shock_sd=0.0, phi=0.0, i_min_frac=0.0, policy_beta=10.0)
        investment, consumption, _ = model.allocation_from_policy(jnp.array([1.0]), model.Y_ss)
        np.testing.assert_allclose(investment, model.I_ss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(consumption, model.C_ss, rtol=1e-5, atol=1e-6)

    def test_safety_clip_caps_investment_below_output(self):
        model = ExactKinkModel(n_sectors=1, phi=0.0, i_min_frac=0.6, hard_floor=True)
        tiny_output = 0.2 * model.Y_ss
        action = jnp.array([8.0])
        self.assertTrue(bool(model.safety_clip_active(action, tiny_output)))
        investment, consumption, _ = model.allocation_from_policy(action, tiny_output)
        np.testing.assert_allclose(investment, 0.9 * tiny_output, rtol=1e-5)
        self.assertGreater(float(jnp.squeeze(consumption)), 0.0)

    def test_state_normalization_uses_requested_scales(self):
        model = ExactKinkModel(n_sectors=1, phi=0.0, i_min_frac=0.8, state_k_scale=0.1, shock_sd=0.02)
        np.testing.assert_allclose(model.state_sd, [0.1, 0.02])
        np.testing.assert_allclose(model._normalize_state(model.K_ss, model.a_ss), jnp.zeros(2), atol=1e-6)

    def test_ss_saving_tail_rises_with_capital(self):
        model = ExactKinkModel(n_sectors=1, phi=0.0, i_min_frac=0.975)
        low = model.ss_saving_tail_value(0.9 * model.K_ss, model.a_ss, n_periods=32)
        high = model.ss_saving_tail_value(1.1 * model.K_ss, model.a_ss, n_periods=32)
        self.assertGreater(float(jnp.squeeze(high)), float(jnp.squeeze(low)))


class NaiveProjectionTest(unittest.TestCase):
    def test_naive_projection_clips_to_the_floor(self):
        model = ProjectedIrreversibleModel(n_sectors=1, phi=0.0, i_min_frac=0.9)
        np.testing.assert_allclose(model.naive_projected_investment(0.5 * model.I_min), model.I_min)
        self.assertEqual(model.policy_map, "saving_rate")


if __name__ == "__main__":
    unittest.main()
