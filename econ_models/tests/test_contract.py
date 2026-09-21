"""Contract checks for the shared Stone-Geary RBC."""

import unittest

from econ_models import StoneGearyRbc, check_algorithm, check_model


class ContractTest(unittest.TestCase):
    def test_stone_geary_passes_every_algorithm(self):
        model = StoneGearyRbc()
        check_model(model)
        for algorithm in ("apg", "deqn", "time_iteration"):
            check_algorithm(model, algorithm)

    def test_utility_is_optional_except_for_apg(self):
        model = StoneGearyRbc()
        model.utility = None
        check_algorithm(model, "deqn")
        check_algorithm(model, "time_iteration")
        with self.assertRaises(ValueError):
            check_algorithm(model, "apg")

    def test_euler_solvers_require_residuals(self):
        model = StoneGearyRbc()
        model.residuals = None
        check_algorithm(model, "apg")
        with self.assertRaises(ValueError):
            check_algorithm(model, "deqn")
        with self.assertRaises(ValueError):
            check_algorithm(model, "time_iteration")

    def test_time_iteration_grids_one_endogenous_state(self):
        model = StoneGearyRbc()
        model.n_endogenous = 2
        with self.assertRaises(ValueError):
            check_algorithm(model, "time_iteration")


if __name__ == "__main__":
    unittest.main()
