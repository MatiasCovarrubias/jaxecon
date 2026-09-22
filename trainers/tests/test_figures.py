"""Shared figures on the Stone-Geary RBC, without a training run."""

import unittest
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt

from econ_models import StoneGearyRbc

from trainers.evaluate import steady_state_policy
from trainers.figures import moment_table, plot_impulses, plot_policies, report, simulate_policies


class FigureTest(unittest.TestCase):
    def setUp(self):
        self.model = StoneGearyRbc()

        def tilt(state):
            gap = state[0] - self.model.state_ss[0]
            return self.model.control_ss + 0.02 * gap

        self.policies = {"steady_state": steady_state_policy(self.model), "tilt": tilt}
        self.config = {
            "episodes": 2,
            "periods": 8,
            "burn": 2,
            "trajectory_periods": 5,
            "irf_periods": 4,
            "ss_horizon": 6,
            "ss_starts": 3,
        }

    def test_policy_slice_includes_low_capital(self):
        fig = plot_policies(self.model, {"steady_state": self.policies["steady_state"]})
        capital = fig.axes[0].lines[0].get_xdata()
        self.assertAlmostEqual(float(capital[0]), 0.2, places=5)
        self.assertAlmostEqual(float(capital[-1]), 2.0, places=5)
        plt.close(fig)

    def test_steady_saving_rate_sits_on_the_steady_state_slice(self):
        fig = plot_policies(self.model, {"steady_state": self.policies["steady_state"]})
        line = fig.axes[0].lines[0].get_ydata()
        self.assertAlmostEqual(float(line[len(line) // 2]), float(self.model.control_ss[0]), places=5)
        plt.close(fig)

    def test_impulse_hits_productivity_one_period_later(self):
        fig = plot_impulses(self.model, self.policies, self.config)
        productivity = fig.axes[1].lines[0].get_ydata()
        self.assertAlmostEqual(float(productivity[0]), 0.0, places=6)
        self.assertAlmostEqual(float(productivity[1]), float(self.model.params.shock_sd), places=6)
        plt.close(fig)

    def test_tables_cover_both_policies(self):
        sample = simulate_policies(self.model, self.policies, self.config)
        text = moment_table(sample)
        self.assertIn("steady_state", text)
        self.assertIn("log saving rate", text)
        self.assertIn("exkurt", text)
        outcome = report(self.model, self.policies, config=self.config)
        self.assertIn("endpoint_sd_%", outcome.stochastic_steady_state)
        for fig in outcome.figures.values():
            plt.close(fig)

    def test_report_writes_the_four_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            report(self.model, self.policies, folder, self.config)
            written = {path.name for path in Path(folder).iterdir()}
        self.assertIn("policy.pdf", written)
        self.assertIn("trajectory.png", written)
        self.assertIn("ergodic.pdf", written)
        self.assertIn("impulse.png", written)
        self.assertIn("moments.txt", written)
        self.assertIn("stochastic_steady_state.txt", written)


if __name__ == "__main__":
    unittest.main()
