"""The shared trainer runs on the Stone-Geary RBC."""

import math
import unittest

from jax import numpy as jnp

from econ_models import StoneGearyRbc

from trainers.apg import steady_saving_tail
from trainers.evaluate import evaluate
from trainers.policy import batch_draw, saving_rate, split_streams
from trainers.time_iteration import interpolated_policy
from trainers.train import train


class SmokeTest(unittest.TestCase):
    def test_ss_saving_tail_matches_discounted_steady_utility(self):
        model = StoneGearyRbc()
        horizon = 4
        utility = model.utility(model.state_ss, model.control_ss)
        weight = sum(model.discount_rate ** t for t in range(horizon))
        tail = steady_saving_tail(model, model.state_ss, horizon)
        self.assertAlmostEqual(float(tail), float(utility * weight), places=5)

    def test_zero_logit_is_the_steady_saving_rate(self):
        model = StoneGearyRbc()
        rate = saving_rate(jnp.zeros((), dtype=model.state_ss.dtype), model)
        self.assertAlmostEqual(float(rate), float(model.control_ss[0]), places=5)

    def test_interpolation_is_linear_off_the_grid(self):
        logk = jnp.array([0.0, 1.0])
        productivity = jnp.array([-0.1, 0.1])
        saving = jnp.array([[0.2, 0.3], [0.4, 0.5]])
        policy = interpolated_policy(logk, productivity, saving, jnp.float32)
        on_node = policy(jnp.array([1.0, 0.1]))
        off_grid = policy(jnp.array([0.5, 0.0]))
        self.assertAlmostEqual(float(on_node[0]), 0.5, places=5)
        self.assertAlmostEqual(float(off_grid[0]), 0.35, places=5)

    def test_same_seed_draws_the_same_training_batch(self):
        model = StoneGearyRbc()
        first = batch_draw(model, split_streams(0)["train"], 2, 3)
        second = batch_draw(model, split_streams(0)["train"], 2, 3)
        other = batch_draw(model, split_streams(1)["train"], 2, 3)
        for left, right in zip(first, second):
            self.assertTrue(jnp.allclose(left, right))
        self.assertFalse(jnp.allclose(first[1], other[1]))

    def test_each_algorithm_returns_finite_metrics(self):
        model = StoneGearyRbc()
        policies = {}
        for algorithm in ("apg", "deqn", "time_iteration"):
            solution = train(model, algorithm)
            policies[algorithm] = solution.policy
            for name, value in solution.metrics.items():
                self.assertTrue(math.isfinite(value), f"{algorithm} {name}={value}")
        evaluated = evaluate(
            model,
            policies,
            {"eval_episodes": 2, "eval_periods": 4, "eval_mc_draws": 2},
        )
        self.assertEqual(set(evaluated), {"steady_state", "apg", "deqn", "time_iteration"})
        self.assertAlmostEqual(evaluated["steady_state"]["ce_vs_ss"], 0.0, places=5)
        for metrics in evaluated.values():
            for name, value in metrics.items():
                self.assertTrue(math.isfinite(value), f"{name}={value}")


if __name__ == "__main__":
    unittest.main()
