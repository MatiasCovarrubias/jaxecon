"""The shared trainer runs on the Stone-Geary RBC."""

import math
import unittest

from econ_models import StoneGearyRbc

from trainers.train import train


class SmokeTest(unittest.TestCase):
    def test_each_algorithm_returns_finite_metrics(self):
        model = StoneGearyRbc()
        for algorithm in ("apg", "deqn", "time_iteration"):
            result = train(model, algorithm)
            self.assertTrue(result)
            for name, value in result.items():
                self.assertTrue(math.isfinite(value), f"{algorithm} {name}={value}")


if __name__ == "__main__":
    unittest.main()
