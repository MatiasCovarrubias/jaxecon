"""Opt-in checks for DEQN residual and inner Monte Carlo diagnostics."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from DEQN.algorithm.loss import (
    plain_deqn_envelope_surrogate,
    plain_deqn_euler_surrogate,
)
from DEQN.diagnostics.theory import (
    evaluate_single_state,
    finite_depth_policy_residual,
)
from DEQN.econ_models.RBC.model import Model


RUN_THEORY_TESTS = os.environ.get("JAXECON_RUN_THEORY_TESTS") == "1"


@unittest.skipUnless(
    RUN_THEORY_TESTS,
    "set JAXECON_RUN_THEORY_TESTS=1 to run theory diagnostics",
)
class TheoryDiagnosticsTest(unittest.TestCase):
    def test_finite_depth_policy_residual_recursion(self):
        euler = jnp.array([1.0, 2.0, 3.0])
        operator = jnp.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            finite_depth_policy_residual(euler, operator, 0.5, 0),
            [1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(
            finite_depth_policy_residual(euler, operator, 0.5, 1),
            [5.0, 9.5, 3.0],
        )
        np.testing.assert_allclose(
            finite_depth_policy_residual(euler, operator, 0.5, 2),
            [20.0, 9.5, 3.0],
        )

    def test_subsistence_cap_zeroes_action_coordinate_mapping(self):
        model = Model(n_sectors=1, shock_sd=0.1, cbar_frac=0.35)
        state = jnp.zeros(model.dim_states, dtype=model.precision)
        policy = jnp.full(
            (model.dim_policies,), 100.0, dtype=model.precision
        )
        mapping = model.action_euler_mapping(state, policy)

        self.assertTrue(bool(jnp.all(mapping.cap_binding)))
        np.testing.assert_allclose(mapping.M, 0.0, atol=0.0)

    def test_action_coordinate_mapping_matches_direct_welfare_derivative(self):
        model = Model(
            n_sectors=1, shock_sd=0.1, cbar_frac=0.35, phi=2.0
        )
        state = jnp.array([0.04, -0.03], dtype=model.precision)
        policy = jnp.array([0.08], dtype=model.precision)
        result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(11),
            mc_draws=32,
            policy_fn=lambda _state: policy,
            residual_kind="levels",
        )
        continuation_costate = result["expected_return"]
        capital, productivity = model._capital_and_productivity(state)
        output = model.production(capital, productivity)

        def one_step_value(action):
            investment, _, _ = model.allocation_from_policy(action, output)
            next_capital = model.next_capital(capital, investment)
            return (
                model.reward(state, action)
                + model.beta
                * jnp.sum(continuation_costate * next_capital)
            )

        direct = jax.grad(one_step_value)(policy)
        np.testing.assert_allclose(
            result["action_euler_residual"],
            direct,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_plain_surrogate_only_differentiates_policy_carrier(self):
        model = Model(n_sectors=1, shock_sd=0.1, cbar_frac=0.35)
        states = jnp.array(
            [[0.02, -0.01], [-0.03, 0.04]], dtype=model.precision
        )
        shocks = jnp.array([[0.2], [-0.15]], dtype=model.precision)
        params = {
            "scale": jnp.array(0.2, dtype=model.precision),
            "bias": jnp.array(0.1, dtype=model.precision),
        }

        def apply_fn(values, obs):
            return values["scale"] * jnp.tanh(
                obs[..., :1] + values["bias"]
            )

        (_, aux), gradient = jax.value_and_grad(
            plain_deqn_euler_surrogate, has_aux=True
        )(params, apply_fn, model, states, shocks)
        manual = jax.grad(
            lambda values: -jnp.mean(
                jnp.sum(aux.e_z * apply_fn(values, states), axis=-1)
            )
        )(params)
        for name in gradient:
            np.testing.assert_allclose(
                gradient[name], manual[name], rtol=1e-6, atol=1e-7
            )

    def test_envelope_surrogate_has_finite_stopped_update(self):
        model = Model(n_sectors=1, shock_sd=0.1, cbar_frac=0.35)
        states = jnp.array(
            [
                [[0.02, -0.01], [-0.01, 0.02]],
                [[-0.03, 0.04], [0.01, -0.02]],
            ],
            dtype=model.precision,
        )
        shocks = jnp.array(
            [
                [[0.2], [-0.1]],
                [[-0.15], [0.1]],
            ],
            dtype=model.precision,
        )
        params = {
            "scale": jnp.array(0.2, dtype=model.precision),
            "bias": jnp.array(0.1, dtype=model.precision),
        }

        def apply_fn(values, obs):
            return values["scale"] * jnp.tanh(
                obs[..., :1] + values["bias"]
            )

        loss, gradient = jax.value_and_grad(
            lambda values: plain_deqn_envelope_surrogate(
                values,
                apply_fn,
                model,
                states,
                shocks,
                depth=1,
            )[0]
        )(params)
        self.assertTrue(np.isfinite(float(loss)))
        for value in gradient.values():
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_envelope_surrogate_differentiates_live_policy_carrier(self):
        model = Model(n_sectors=1, shock_sd=0.1, cbar_frac=0.35)
        states = jnp.array(
            [
                [[0.02, -0.01], [-0.01, 0.02]],
                [[-0.03, 0.04], [0.01, -0.02]],
            ],
            dtype=model.precision,
        )
        shocks = jnp.array(
            [
                [[0.2], [-0.1]],
                [[-0.15], [0.1]],
            ],
            dtype=model.precision,
        )
        params = {
            "scale": jnp.array(0.2, dtype=model.precision),
            "bias": jnp.array(0.1, dtype=model.precision),
        }

        def apply_fn(values, obs):
            return values["scale"] * jnp.tanh(
                obs[..., :1] + values["bias"]
            )

        _, gradient = jax.value_and_grad(
            lambda values: plain_deqn_envelope_surrogate(
                values,
                apply_fn,
                model,
                states,
                shocks,
                depth=1,
            )[0]
        )(params)
        self.assertGreater(
            float(jnp.linalg.norm(jax.flatten_util.ravel_pytree(gradient)[0])),
            0.0,
        )

    def test_deterministic_no_shock_consistency(self):
        model = Model(n_sectors=1, shock_sd=0.0, cbar_frac=0.4)
        state = jnp.zeros(model.dim_states, dtype=model.precision)
        policy_fn = lambda obs: jnp.zeros(
            model.dim_policies, dtype=obs.dtype
        )

        result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(0),
            mc_draws=4,
            policy_fn=policy_fn,
            residual_kind="ratio",
        )

        np.testing.assert_allclose(
            result["next_states"],
            jnp.broadcast_to(state, result["next_states"].shape),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result["per_draw_expectation"],
            jnp.broadcast_to(
                result["per_draw_expectation"][0],
                result["per_draw_expectation"].shape,
            ),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(result["ratio_residual"], 0.0, atol=1e-5)
        np.testing.assert_allclose(result["levels_residual"], 0.0, atol=1e-5)

    def test_outputs_have_draw_shapes_and_finite_values(self):
        model = Model(n_sectors=1, shock_sd=0.15, cbar_frac=0.4)
        state = jnp.array([0.03, -0.02], dtype=model.precision)
        policy_fn = lambda obs: 0.2 * jnp.tanh(obs[:1])

        result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(1),
            mc_draws=6,
            policy_fn=policy_fn,
            residual_kind="levels",
        )

        self.assertEqual(result["shocks"].shape, (6, 1))
        self.assertEqual(result["next_states"].shape, (6, 2))
        self.assertEqual(result["per_draw_expectation"].shape, (6, 1))
        self.assertEqual(result["per_draw_levels_residual"].shape, (6, 1))
        self.assertEqual(result["levels_residual"].shape, (1,))
        self.assertTrue(
            all(np.isfinite(np.asarray(value)).all() for value in result.values())
        )

    def test_levels_are_affine_but_ratio_is_not(self):
        model = Model(n_sectors=1, shock_sd=0.2, cbar_frac=0.4)
        state = jnp.array([-0.04, 0.03], dtype=model.precision)
        policy_fn = lambda obs: 0.25 * jnp.tanh(obs[:1] + 0.1)

        result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(2),
            mc_draws=8,
            policy_fn=policy_fn,
            residual_kind="levels",
            stop_gradient=True,
        )

        np.testing.assert_allclose(
            result["levels_residual"],
            jnp.mean(result["per_draw_levels_residual"], axis=0),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertGreater(
            float(jnp.std(result["per_draw_expectation"])), 1e-5
        )
        self.assertFalse(
            np.allclose(
                np.asarray(result["ratio_residual"]),
                np.asarray(
                    jnp.mean(result["per_draw_ratio_residual"], axis=0)
                ),
                rtol=1e-6,
                atol=1e-7,
            )
        )

    def test_stop_and_full_gradients_are_finite_and_distinct(self):
        model = Model(n_sectors=1, shock_sd=0.2, cbar_frac=0.4)
        state = jnp.array([0.04, -0.03], dtype=model.precision)
        params = {
            "scale": jnp.array(0.2, dtype=model.precision),
            "bias": jnp.array(0.1, dtype=model.precision),
        }

        def apply_fn(parameter_values, obs):
            return parameter_values["scale"] * jnp.tanh(
                obs[:1] + parameter_values["bias"]
            )

        stop_result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(3),
            mc_draws=8,
            params=params,
            apply_fn=apply_fn,
            residual_kind="levels",
            stop_gradient=True,
            return_gradient=True,
        )
        full_result = evaluate_single_state(
            model,
            state,
            random.PRNGKey(3),
            mc_draws=8,
            params=params,
            apply_fn=apply_fn,
            residual_kind="levels",
            stop_gradient=False,
            return_gradient=True,
        )

        for result in (stop_result, full_result):
            self.assertTrue(np.isfinite(float(result["loss"])))
            for value in result["gradient"].values():
                self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertFalse(
            np.allclose(
                np.asarray(stop_result["gradient"]["scale"]),
                np.asarray(full_result["gradient"]["scale"]),
                rtol=1e-6,
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()
