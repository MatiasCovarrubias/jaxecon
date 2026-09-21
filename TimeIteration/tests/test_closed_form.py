"""Stage 0 gate: log utility and ? = 1 imply a saving rate of exactly ??."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from TimeIteration.algorithm.implicit_diff import implicit_policy_fn
from TimeIteration.algorithm.solve import build_grids, solve
from TimeIteration.diagnostics import allocation_on_grid
from TimeIteration.models.protocol import GridSpec
from TimeIteration.models.rbc import RbcModel, closed_form_params


def _spec():
    return GridSpec(n_a=5, n_k=15, k_min_rel=0.75, k_max_rel=1.3)


def _perturbed_start(model, params, spec, scale=0.5):
    grids = build_grids(model, params, spec)
    x0 = model.initial_policy(params, grids)
    x0 = x0.at[..., 0].set(scale * x0[..., 0])
    return x0


class ClosedFormTest(unittest.TestCase):
    def test_log_utility_full_depreciation_saving_rate(self):
        model = RbcModel(unknown="saving_rate")
        params = closed_form_params()
        spec = _spec()
        sol = solve(model, params, spec, x0=_perturbed_start(model, params, spec), tol=1e-12, max_iter=80)
        saving = allocation_on_grid(model, params, sol.grids, sol.x_grid).s
        target = params.alpha * params.beta
        np.testing.assert_allclose(saving, target, atol=1e-12, rtol=0.0)
        self.assertLess(float(sol.info.residual_norm), 1e-11)

    def test_slack_irreversibility_recovers_the_same_policy(self):
        model = RbcModel(irreversible=True, unknown="saving_rate")
        params = closed_form_params(i_min_frac=0.01)
        spec = _spec()
        sol = solve(model, params, spec, x0=_perturbed_start(model, params, spec), tol=1e-12, max_iter=80)
        aux = allocation_on_grid(model, params, sol.grids, sol.x_grid)
        target = params.alpha * params.beta
        np.testing.assert_allclose(aux.s, target, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(aux.mu, 0.0, atol=1e-12)
        self.assertTrue(bool(jnp.all(aux.I > params.i_min_frac * sol.grids.ss.I)))

    def test_implicit_derivative_of_the_saving_rate(self):
        model = RbcModel(unknown="saving_rate")
        params = closed_form_params()
        spec = GridSpec(n_a=3, n_k=7, k_min_rel=0.8, k_max_rel=1.25)
        x0 = _perturbed_start(model, params, spec)
        policy_fn = implicit_policy_fn(model, spec, x0, tol=1e-12, max_iter=80, newton_steps=20)

        def mean_saving(p):
            return jnp.mean(policy_fn(p)[..., 0])

        grad = jax.grad(mean_saving)(params)
        np.testing.assert_allclose(float(grad.alpha), float(params.beta), atol=1e-6)
        np.testing.assert_allclose(float(grad.beta), float(params.alpha), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
