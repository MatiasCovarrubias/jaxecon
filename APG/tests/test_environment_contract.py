"""The APG environment contract, checked on every RBC variant, plus wrapper-specific behaviour."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from APG.environments import RbcMultiSector, check_environment

VARIANTS = {
    "smooth": dict(N=1),
    "two_sector": dict(N=2),
    "projected_irreversible": dict(N=1, phi=0.0, i_min_frac=0.975),
    "penalized_irreversible": dict(
        N=1, phi=0.0, i_min_frac=0.975, project_investment=False, investment_penalty=10.0
    ),
    "exact_kink": dict(N=1, phi=0.0, i_min_frac=0.8, policy_map="exact_kink"),
    "exact_kink_hard_floor": dict(N=1, phi=0.0, i_min_frac=0.8, policy_map="exact_kink", hard_floor=True),
}


class EnvironmentContractTest(unittest.TestCase):
    def test_every_rbc_variant_satisfies_the_contract(self):
        for name, kwargs in VARIANTS.items():
            with self.subTest(variant=name):
                check_environment(RbcMultiSector(**kwargs))

    def test_checker_rejects_a_broken_environment(self):
        env = RbcMultiSector(N=1)
        env.transition = lambda state, action, shock: state + 1.0
        with self.assertRaisesRegex(AssertionError, "fixed point"):
            check_environment(env)


class SavingRateWrapperTest(unittest.TestCase):
    def test_extreme_actions_keep_consumption_and_investment_positive(self):
        env = RbcMultiSector(N=1)
        output = jnp.array([0.5])
        for action in (jnp.full(1, -1e6), jnp.full(1, 1e6)):
            investment, consumption, saving_rate = env.allocation_from_action(action, output)
            self.assertTrue(bool(jnp.all(investment > 0)))
            self.assertTrue(bool(jnp.all(consumption > 0)))
            self.assertTrue(bool(jnp.all(saving_rate >= env.saving_rate_min)))
            self.assertTrue(bool(jnp.all(saving_rate <= env.saving_rate_max)))
            np.testing.assert_allclose(investment + consumption, output, rtol=1e-6, atol=1e-7)

    def test_invalid_saving_rate_bounds_are_rejected(self):
        for lower, upper in ((0.0, 0.9), (0.1, 1.0), (0.8, 0.2)):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                RbcMultiSector(saving_rate_min=lower, saving_rate_max=upper)

    def test_step_returns_gym_style_tuple(self):
        env = RbcMultiSector(N=1, shock_sd=0.0)
        state = env.initial_state(random.PRNGKey(0), 0)
        next_obs, next_state, reward, done, _ = env.step(
            random.PRNGKey(1), state, env.deterministic_steady_state_action()
        )
        np.testing.assert_allclose(next_obs, next_state)
        np.testing.assert_allclose(next_obs, state, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(reward, env.reward_ss, rtol=1e-5, atol=1e-6)
        self.assertFalse(bool(done))


class InvestmentPenaltyTest(unittest.TestCase):
    def test_positive_penalty_requires_unprojected_policy(self):
        with self.assertRaises(ValueError):
            RbcMultiSector(N=1, phi=0.0, i_min_frac=0.975, investment_penalty=1.0, project_investment=True)

    def test_penalty_lowers_reward_and_pushes_saving_up_when_violating(self):
        env = RbcMultiSector(N=1, phi=0.0, i_min_frac=0.975, investment_penalty=10.0, project_investment=False)
        state = env.econ._normalize_state(0.5 * env.K_ss, env.a_ss)
        action = jnp.zeros(env.action_dim)

        raw_reward = env.econ.reward(state, action)
        training_reward = env.training_reward(state, action)
        reward_gradient = jax.grad(lambda a: env.training_reward(state, a).sum())(action)

        self.assertLess(float(training_reward), float(raw_reward))
        self.assertGreater(float(jnp.squeeze(reward_gradient)), 0.0)


if __name__ == "__main__":
    unittest.main()
