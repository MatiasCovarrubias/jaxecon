"""Tests for the opt-in frozen-policy gradient diagnostic."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from APG.diagnostics import (
    finite_horizon_return,
    policy_gradient_diagnostic,
    statewise_action_value_gradient,
)
from APG.environments import RbcMultiSector


def _tree_assert_allclose(test_case, left, right, **kwargs):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    test_case.assertEqual(len(left_leaves), len(right_leaves))
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(left_leaf, right_leaf, **kwargs)


def _simple_case():
    params = {
        "bias": jnp.array([0.1]),
        "slope": jnp.array([0.25]),
    }
    policy_fn = lambda p, state: p["bias"] + p["slope"] * state
    reward_fn = lambda state, action: -jnp.sum(state**2 + 0.5 * action**2)
    transition_fn = lambda state, action, shock: 0.8 * state + action + shock
    return (
        jnp.array([0.2]),
        jnp.array([[0.0], [0.1], [-0.05]]),
        params,
        policy_fn,
        reward_fn,
        transition_fn,
    )


class TheoryDiagnosticApiTest(unittest.TestCase):
    def test_diagnostic_supports_pytree_parameters(self):
        diagnostic = policy_gradient_diagnostic(
            *_simple_case(),
            discount_rate=0.9,
            epsilon=1e-3,
        )

        self.assertEqual(diagnostic.return_value.shape, ())
        self.assertEqual(set(diagnostic.autodiff_gradient), {"bias", "slope"})
        _tree_assert_allclose(
            self,
            diagnostic.autodiff_gradient,
            diagnostic.forward_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
        _tree_assert_allclose(
            self,
            diagnostic.autodiff_gradient,
            diagnostic.reverse_gradient,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_statewise_pullback_reconstructs_gradient_with_terminal_value(self):
        (
            initial_state,
            shocks,
            params,
            policy_fn,
            reward_fn,
            transition_fn,
        ) = _simple_case()
        terminal_value_fn = lambda state: -0.3 * jnp.sum(state**2)
        statewise = statewise_action_value_gradient(
            initial_state,
            shocks,
            params,
            policy_fn,
            reward_fn,
            transition_fn,
            0.9,
            terminal_value_fn=terminal_value_fn,
        )
        autodiff = jax.grad(
            lambda values: finite_horizon_return(
                initial_state,
                shocks,
                values,
                policy_fn,
                reward_fn,
                transition_fn,
                0.9,
                terminal_value_fn=terminal_value_fn,
            )
        )(params)

        self.assertEqual(statewise.action_value_gradients.shape, (3, 1))
        _tree_assert_allclose(
            self,
            statewise.parameter_gradient,
            autodiff,
            rtol=1e-5,
            atol=1e-6,
        )


@unittest.skipUnless(
    os.environ.get("JAXECON_RUN_THEORY_TESTS") == "1",
    "set JAXECON_RUN_THEORY_TESTS=1 to run the RBC theory diagnostic",
)
class RbcTheoryDiagnosticTest(unittest.TestCase):
    def test_smooth_rbc_gradients_agree(self):
        env = RbcMultiSector(N=1, shock_sd=0.01, phi=0.5)
        initial_state = jnp.array([0.03, -0.02], dtype=env.precision)
        shocks = jnp.array(
            [[0.10], [-0.15], [0.05], [0.02], [-0.08]],
            dtype=env.precision,
        )
        params = {
            "bias": jnp.array([0.02], dtype=env.precision),
            "weight": jnp.array([[0.04, -0.03]], dtype=env.precision),
        }

        def policy_fn(policy_params, state):
            return policy_params["bias"] + policy_params["weight"] @ state

        diagnostic = policy_gradient_diagnostic(
            initial_state,
            shocks,
            params,
            policy_fn,
            env.training_reward,
            env.transition,
            env.discount_rate,
            epsilon=2e-3,
        )

        _tree_assert_allclose(
            self,
            diagnostic.autodiff_gradient,
            diagnostic.forward_gradient,
            rtol=3e-4,
            atol=3e-5,
        )
        _tree_assert_allclose(
            self,
            diagnostic.autodiff_gradient,
            diagnostic.reverse_gradient,
            rtol=3e-4,
            atol=3e-5,
        )
        _tree_assert_allclose(
            self,
            diagnostic.autodiff_gradient,
            diagnostic.finite_difference_gradient,
            rtol=4e-3,
            atol=4e-4,
        )


if __name__ == "__main__":
    unittest.main()
