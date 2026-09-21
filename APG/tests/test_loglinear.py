"""In-house log-linear policy: LQ approximation and Riccati feedback."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import random

from APG.environments import RbcMultiSector
from APG.loglinear import build_actor, lq_design, prepare_loglinear, solve_loglinear, solve_riccati
from APG.loglinear.design import apply_lq_design
from APG.neural_nets import PolicyNetLoglinear

jax_config.update("jax_enable_x64", True)


def _env(**kwargs):
    values = dict(
        N=1,
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        rho=0.7,
        shock_sd=0.02,
        phi=2.0,
        eps_c=0.5,
        double_precision=True,
        precision=jnp.float64,
    )
    values.update(kwargs)
    return RbcMultiSector(**values)


def _simulated_sd(sol, rng, n_periods=40000, burnin=2000):
    A, B, C, D = (jnp.asarray(m) for m in (sol.A, sol.B, sol.C, sol.D))

    def period(state, key):
        shock = random.normal(key, shape=(D.shape[1],), dtype=A.dtype)
        action = C @ state
        next_state = A @ state + B @ action + D @ shock
        return next_state, (next_state, action)

    _, (states, actions) = jax.lax.scan(
        period, jnp.zeros(A.shape[0], dtype=A.dtype), random.split(rng, n_periods + burnin)
    )
    return states[burnin:].std(axis=0), actions[burnin:].std(axis=0)


class LogLinearSolveTest(unittest.TestCase):
    def test_standard_rbc_feedback_and_stability(self):
        env = _env()
        state_sd0, policies_sd0 = env.state_sd, env.policies_sd
        sol = solve_loglinear(env)

        np.testing.assert_allclose(env.state_sd, state_sd0)
        np.testing.assert_allclose(env.policies_sd, policies_sd0)
        self.assertEqual(sol.C.shape, (1, 2))
        self.assertLess(sol.spectral_radius, 1.0)
        np.testing.assert_allclose(sol.foc_residual, 0.0, atol=1e-6)
        self.assertLess(sol.C[0, 0], 0.0)
        self.assertGreater(sol.C[0, 1], 0.0)
        self.assertTrue(np.all(sol.states_sd > 0))
        self.assertTrue(np.all(sol.policies_sd > 0))
        self.assertTrue(np.all(np.linalg.eigvalsh(sol.R) < 0))

    def test_scipy_and_iteration_agree(self):
        env = _env()
        sol = solve_loglinear(env, solver="scipy")
        P, C = solve_riccati(
            sol.A, sol.B, sol.Q, sol.R, sol.W, float(env.discount_rate), solver="iterate"
        )
        np.testing.assert_allclose(C, sol.C, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(P, sol.P, rtol=1e-5, atol=1e-6)

    def test_riccati_fixed_point(self):
        env = _env()
        sol = solve_loglinear(env)
        beta = float(env.discount_rate)
        A, B, P, Q, R, W = sol.A, sol.B, sol.P, sol.Q, sol.R, sol.W
        cross = W + beta * A.T @ P @ B
        rhs = Q + beta * A.T @ P @ A - cross @ np.linalg.solve(R + beta * B.T @ P @ B, cross.T)
        np.testing.assert_allclose(P, rhs, rtol=1e-9, atol=1e-9)

    def test_lyapunov_sds_match_linear_simulation(self):
        env = _env()
        sol = solve_loglinear(env)
        sim_state_sd, sim_policy_sd = _simulated_sd(sol, random.PRNGKey(0))
        np.testing.assert_allclose(sim_state_sd, sol.states_sd, rtol=0.15, atol=0.002)
        np.testing.assert_allclose(sim_policy_sd, sol.policies_sd, rtol=0.15, atol=0.002)

    def test_two_sector_shapes(self):
        env = _env(N=2)
        sol = solve_loglinear(env)
        self.assertEqual(sol.C.shape, (2, 4))
        self.assertEqual(sol.states_sd.shape, (4,))
        self.assertEqual(sol.policies_sd.shape, (2,))
        self.assertLess(sol.spectral_radius, 1.0)

    def test_zero_shock_falls_back_to_unit_scales(self):
        env = _env(shock_sd=0.0)
        sol = solve_loglinear(env)
        np.testing.assert_allclose(sol.states_sd, 1.0)
        np.testing.assert_allclose(sol.policies_sd, 1.0)
        self.assertLess(sol.spectral_radius, 1.0)

    def test_convex_return_is_rejected(self):
        env = _env()
        sol = solve_loglinear(env)
        with self.assertRaises(ValueError):
            solve_riccati(sol.A, sol.B, sol.Q, -sol.R, sol.W, float(env.discount_rate))


class LogLinearBaselineTest(unittest.TestCase):
    def test_untrained_network_is_the_linear_policy(self):
        env = _env()
        sol = solve_loglinear(env)
        net = PolicyNetLoglinear(
            features=[8],
            n_out=env.action_dim,
            C=jnp.asarray(sol.C),
            states_sd=jnp.ones(2),
            policies_sd=jnp.ones(1),
            precision=jnp.float64,
        )
        params = net.init(random.PRNGKey(0), env.obs_ss)
        state = jnp.array([0.04, -0.02], dtype=jnp.float64)
        np.testing.assert_allclose(net.apply(params, state), sol.C @ state, atol=1e-12)

    def test_prepare_normalizes_env_and_builds_zscore_net(self):
        env = _env()
        config = {
            "loglinear_baseline": True,
            "loglinear_normalize": True,
            "layers": [4],
            "use_terminal_value": False,
        }
        sol = prepare_loglinear(env, config)
        np.testing.assert_allclose(env.state_sd, sol.states_sd)
        np.testing.assert_allclose(env.policies_sd, sol.policies_sd)
        self.assertEqual(config["loglinear_C"], sol.C.tolist())
        net = build_actor(env, config, jnp.float64, solution=sol)
        params = net.init(random.PRNGKey(1), jnp.zeros_like(env.obs_ss))
        zscore = jnp.array([1.0, 0.0], dtype=jnp.float64)
        expected = (sol.C @ (zscore * sol.states_sd)) / sol.policies_sd
        np.testing.assert_allclose(net.apply(params, zscore), expected, atol=1e-12)

    def test_prepare_is_a_no_op_when_disabled(self):
        env = _env()
        self.assertIsNone(prepare_loglinear(env, {"loglinear_baseline": False}))
        np.testing.assert_allclose(env.state_sd, 1.0)

    def test_normalization_keeps_steady_state_fixed_point(self):
        env = _env()
        sol = solve_loglinear(env)
        env.set_scales(sol.states_sd, sol.policies_sd)
        state = env.initial_state(random.PRNGKey(0), 0)
        action = env.deterministic_steady_state_action()
        shock = jnp.zeros_like(env.sample_shock(random.PRNGKey(1)))
        np.testing.assert_allclose(state, 0.0, atol=1e-12)
        np.testing.assert_allclose(env.transition(state, action, shock), state, atol=1e-10)


class LqDesignTest(unittest.TestCase):
    def test_design_matches_solve_and_prints_stability(self):
        env = _env()
        sol = solve_loglinear(env)
        design = lq_design(env)
        np.testing.assert_allclose(design.C, sol.C)
        np.testing.assert_allclose(design.P, sol.P)
        np.testing.assert_allclose(design.A_closed, sol.A + sol.B @ sol.C)
        self.assertEqual(design.Sigma.shape, sol.A.shape)
        resid = design.A_closed @ design.Sigma @ design.A_closed.T + design.D @ design.D.T - design.Sigma
        np.testing.assert_allclose(resid, 0.0, atol=1e-8)

    def test_lq_terminal_value_at_steady_state(self):
        env = _env()
        design = lq_design(env)
        config = {}
        apply_lq_design(env, env.econ, config, design=design)
        P = jnp.asarray(config["lq_terminal_P"])
        costate = jnp.asarray(config["lq_terminal_costate"])

        def W(x):
            return jnp.dot(costate, x) + 0.5 * jnp.dot(x, P @ x)

        ss = jnp.zeros_like(env.obs_ss)
        np.testing.assert_allclose(W(ss), 0.0, atol=1e-12)
        np.testing.assert_allclose(jax.grad(W)(ss), costate, atol=1e-12)
        xhat = jnp.array([0.03, -0.02], dtype=jnp.float64)
        np.testing.assert_allclose(
            W(xhat) - jnp.dot(costate, xhat),
            0.5 * jnp.dot(xhat, P @ xhat),
            atol=1e-8,
        )

    def test_stationary_start_ignores_init_range(self):
        env = _env()
        design = lq_design(env)
        apply_lq_design(env, env.econ, {}, design=design)
        draw = env.initial_state(random.PRNGKey(4), 50)
        box = env.initial_state(random.PRNGKey(4), 50, mode="box")
        self.assertFalse(np.allclose(draw, box))
        self.assertEqual(draw.shape, env.obs_ss.shape)

    def test_knobs_are_independent(self):
        from APG.loglinear.design import inject_lq_design

        env = _env()
        design = lq_design(env)
        unit_sd = np.asarray(env.state_sd)
        config = inject_lq_design(
            {},
            design,
            residual=False,
            scales=False,
            stationary_start=True,
            apg_tail=True,
        )
        apply_lq_design(env, env.econ, config, design=design)
        np.testing.assert_allclose(env.state_sd, unit_sd)
        self.assertEqual(env.econ.initial_state_mode, "stationary")
        self.assertFalse(config["loglinear_baseline"])
        self.assertTrue(config["use_lq_terminal_value"])


if __name__ == "__main__":
    unittest.main()
