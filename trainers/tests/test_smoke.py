"""The shared trainer runs on the Stone-Geary RBC."""

import math
import unittest

from econ_models import StoneGearyRbc

from trainers.evaluate import compare
from trainers.train import train


class SmokeTest(unittest.TestCase):
    def test_each_algorithm_returns_finite_metrics(self):
        model = StoneGearyRbc()
        policies = {}
        for algorithm in ("apg", "deqn", "time_iteration"):
            solution = train(model, algorithm)
            policies[algorithm] = solution.policy
            for name, value in solution.metrics.items():
                self.assertTrue(math.isfinite(value), f"{algorithm} {name}={value}")
        evaluated = compare(
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
