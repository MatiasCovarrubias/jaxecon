import unittest

import jax
import jax.numpy as jnp
import numpy as np

from TimeIteration.algorithm.interpolation import interp1d, interp_policy
from TimeIteration.algorithm.newton import newton_solve
from TimeIteration.algorithm.solve import solve
from TimeIteration.diagnostics import allocation_on_grid, euler_errors_on_grid, simulate
from TimeIteration.models.protocol import GridSpec
from TimeIteration.models.rbc import RbcModel, default_params


class InterpolationTest(unittest.TestCase):
    def test_linear_function_is_recovered_with_extrapolation(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        fp = 2.0 * xp + 1.0
        query = jnp.array([-0.25, 0.0, 0.37, 1.0, 1.5])
        values = jax.vmap(lambda q: interp1d(q, xp, fp))(query)
        np.testing.assert_allclose(values, 2.0 * query + 1.0, atol=1e-12)


class NewtonTest(unittest.TestCase):
    def test_damped_newton_solves_a_small_system(self):
        def F(x):
            return jnp.stack([x[0] ** 2 - 1.0, x[1] - 2.0])

        root = newton_solve(F, jnp.array([0.4, 0.0]), n_steps=20)
        np.testing.assert_allclose(root, jnp.array([1.0, 2.0]), atol=1e-10)


class SubsistenceTest(unittest.TestCase):
    def test_cbar_zero_matches_crra_and_does_not_shift_ss(self):
        model = RbcModel()
        plain = default_params()
        with_zero = default_params(cbar_frac=0.0)
        ss_plain = model.steady_state(plain)
        ss_zero = model.steady_state(with_zero)
        np.testing.assert_allclose(ss_zero.K, ss_plain.K)
        np.testing.assert_allclose(ss_zero.C, ss_plain.C)
        np.testing.assert_allclose(ss_zero.u_c, ss_plain.C ** (-plain.sigma))

    def test_cbar_shifts_marginal_utility_not_steady_state(self):
        from TimeIteration.models.rbc import marginal_utility

        model = RbcModel()
        params = default_params(cbar_frac=0.6)
        ss = model.steady_state(params)
        ss0 = model.steady_state(default_params(cbar_frac=0.0))
        np.testing.assert_allclose(ss.K, ss0.K)
        np.testing.assert_allclose(ss.C, ss0.C)
        c_bar = 0.6 * ss.C
        np.testing.assert_allclose(ss.u_c, marginal_utility(ss.C, params.sigma, c_bar))
        self.assertGreater(float(ss.u_c), float(ss0.u_c))


class ResidualTest(unittest.TestCase):
    def test_deterministic_steady_state_clears_the_euler(self):
        model = RbcModel()
        params = default_params(shock_sd=0.0)
        ss = model.steady_state(params)
        x = jnp.array([ss.I])
        Ex = model.expectand(params, jnp.asarray(0.0), ss.K, x, ss)
        residual = model.arbitrage(params, jnp.asarray(0.0), ss.K, x, Ex, ss)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_policy_interpolation_is_componentwise(self):
        K = jnp.array([1.0, 2.0, 3.0])
        x_at_a = jnp.stack([K, 10.0 * K], axis=-1)
        value = interp_policy(K, x_at_a, jnp.asarray(1.5))
        np.testing.assert_allclose(value, jnp.array([1.5, 15.0]), atol=1e-12)


class IrreversibleTest(unittest.TestCase):
    def test_high_floor_binds_everywhere(self):
        model = RbcModel(irreversible=True)
        params = default_params(i_min_frac=1.2, shock_sd=0.0, rho=0.0)
        spec = GridSpec(n_a=1, n_k=11, k_min_rel=0.9, k_max_rel=1.1)
        sol = solve(
            model,
            params,
            spec,
            tol=1e-10,
            max_iter=40,
            newton_steps=20,
            anderson_memory=5,
        )
        aux = allocation_on_grid(model, params, sol.grids, sol.x_grid)
        I_min = params.i_min_frac * sol.grids.ss.I
        np.testing.assert_allclose(aux.I, I_min, atol=1e-8)
        self.assertTrue(bool(jnp.all(aux.mu > 0.0)))
        residual = euler_errors_on_grid(model, params, sol.grids, sol.x_grid)
        self.assertLess(float(jnp.max(jnp.abs(residual))), 1e-7)
        traj = simulate(model, params, sol.grids, sol.x_grid, jax.random.PRNGKey(0), T=8, N=4)
        self.assertEqual(traj.C.shape, (4, 8))
        self.assertTrue(bool(jnp.all(jnp.isfinite(traj.C))))
        self.assertTrue(bool(jnp.all(traj.I >= I_min - 1e-10)))


if __name__ == "__main__":
    unittest.main()
