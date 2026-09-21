"""The APG rollout and its agreement with the DEQN rollout on the shared RBC."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import random

from APG.algorithm.simulation import create_episode_simul_fn as create_apg_simul_fn
from APG.algorithm.welfare import create_welfare_fn
from APG.environments import RbcMultiSector
from APG.neural_nets import PolicyNet
from DEQN.algorithm.simulation import create_episode_simul_fn as create_deqn_simul_fn
from DEQN.algorithm.simulation import rollout_episode_obs, sample_episode_shocks
from DEQN.econ_models.RBC.model import Model
from DEQN.econ_models.RBC.train_shared import split_shared_rbc_rngs
from DEQN.econ_models.RBC.welfare_eval import create_welfare_eval_fn

BASE_CONFIG = {
    "periods_per_epis": 8,
    "init_range": 5,
    "simul_vol_scale": 1.0,
    "use_terminal_value": False,
}


def _shared_setup(features=(8, 4)):
    model = Model(n_sectors=1, shock_sd=0.02, phi=2.0)
    env = RbcMultiSector(N=1, shock_sd=0.02, phi=2.0)
    net = PolicyNet(features=list(features), n_out=model.dim_policies)
    params = net.init(random.PRNGKey(0), jnp.zeros_like(model.state_ss))
    ts = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(0.001))
    return model, env, net, params, ts


class DeqnApgSharedConventionTest(unittest.TestCase):
    def test_state_and_policy_normalization_match(self):
        model, env, *_ = _shared_setup()
        np.testing.assert_allclose(model.state_ss, env.obs_ss)
        np.testing.assert_allclose(model.state_sd, env.obs_sd)
        np.testing.assert_allclose(model.policies_ss, env.policy_ss)
        np.testing.assert_allclose(model.policies_sd, env.policy_sd)
        self.assertEqual(model.dim_states, env.obs_dim)
        self.assertEqual(model.dim_policies, env.action_dim)

    def test_rollouts_share_shocks_and_next_states(self):
        model, env, _, params, ts = _shared_setup()
        deqn_obs = create_deqn_simul_fn(model, BASE_CONFIG)(ts, random.PRNGKey(7))
        _, trajectory, _ = create_apg_simul_fn(env, BASE_CONFIG)(params, ts, random.PRNGKey(7))
        np.testing.assert_allclose(deqn_obs, trajectory.obs, rtol=1e-5, atol=1e-6)

    def test_shared_seed_keeps_training_streams_aligned(self):
        deqn = split_shared_rbc_rngs(6)
        apg = split_shared_rbc_rngs(6, extra_names=("euler",))
        leftover, rng_init, rng_epoch, rng_eval, rng_welfare = random.split(
            random.PRNGKey(6), 5
        )
        leftover, rng_euler = random.split(leftover)
        for name, key in (
            ("init", rng_init),
            ("epoch", rng_epoch),
            ("eval", rng_eval),
            ("welfare", rng_welfare),
        ):
            np.testing.assert_array_equal(deqn[name], key)
            np.testing.assert_array_equal(apg[name], key)
        np.testing.assert_array_equal(apg["euler"], rng_euler)
        self.assertFalse(bool(jnp.all(apg["euler"] == deqn["epoch"])))

    def test_same_params_give_same_welfare(self):
        model, env, net, params, _ = _shared_setup()
        kwargs = dict(policy_fn=lambda p, obs: net.apply(p, obs), horizon=16, n_epis=8, init_range=0)
        deqn_metrics = create_welfare_eval_fn(model, **kwargs)(params, random.PRNGKey(3))
        apg_metrics = create_welfare_eval_fn(env.econ, **kwargs)(params, random.PRNGKey(3))
        np.testing.assert_allclose(deqn_metrics.welfare, apg_metrics.welfare)
        np.testing.assert_allclose(deqn_metrics.ce_vs_fixed_saving_rate, apg_metrics.ce_vs_fixed_saving_rate)


class ApgRolloutTest(unittest.TestCase):
    def test_antithetic_episodes_average_opposite_shock_paths(self):
        model, env, _, params, ts = _shared_setup()
        key = random.PRNGKey(7)
        shocks = sample_episode_shocks(model, key, BASE_CONFIG["periods_per_epis"], 1.0)
        init_obs = model.initial_state(key, BASE_CONFIG["init_range"])

        ret_plus, traj_plus, _ = create_apg_simul_fn(env, {**BASE_CONFIG, "antithetic_episodes": False})(params, ts, key)
        ret_pair, traj_pair, _ = create_apg_simul_fn(env, {**BASE_CONFIG, "antithetic_episodes": True})(params, ts, key)
        np.testing.assert_allclose(traj_pair.obs, traj_plus.obs, rtol=1e-5, atol=1e-6)

        obs_minus = rollout_episode_obs(model, ts, init_obs, -shocks)
        ret_minus, discount, obs = 0.0, 1.0, init_obs
        for shock in np.asarray(-shocks):
            action = ts.apply_fn(params, obs)
            ret_minus += discount * float(model.reward(obs, action))
            obs = model.step(obs, action, shock)
            discount *= float(env.discount_rate)
        np.testing.assert_allclose(obs, obs_minus[-1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(float(ret_pair), 0.5 * (float(ret_plus) + ret_minus), rtol=1e-5, atol=1e-6)

    def test_model_terminal_value_enters_return_and_gradient(self):
        env = RbcMultiSector(N=1, shock_sd=0.0, phi=0.0)
        net = PolicyNet(features=[4], n_out=env.action_dim)
        params = net.init(random.PRNGKey(0), env.obs_ss)
        ts = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(0.001))
        horizon = 7
        config = {
            "periods_per_epis": 2,
            "init_range": 0,
            "simul_vol_scale": 0.0,
            "use_model_terminal_value": True,
            "terminal_value_horizon": horizon,
        }
        with_tail = create_apg_simul_fn(env, config)
        without_tail = create_apg_simul_fn(env, {**config, "use_model_terminal_value": False})

        _, trajectory, terminal = with_tail(params, ts, random.PRNGKey(1))
        np.testing.assert_allclose(terminal, env.terminal_value(trajectory.obs[-1], horizon), rtol=1e-6)

        grad_with = jax.grad(lambda p: with_tail(p, ts, random.PRNGKey(1))[0])(params)
        grad_without = jax.grad(lambda p: without_tail(p, ts, random.PRNGKey(1))[0])(params)
        difference = jax.tree_util.tree_map(lambda a, b: jnp.sum(jnp.square(a - b)), grad_with, grad_without)
        self.assertGreater(float(sum(jax.tree_util.tree_leaves(difference))), 0.0)

    def test_two_terminal_bootstraps_are_rejected(self):
        env = RbcMultiSector(N=1)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            create_apg_simul_fn(
                env,
                {"periods_per_epis": 2, "use_terminal_value": True, "use_model_terminal_value": True},
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            create_apg_simul_fn(
                env,
                {
                    "periods_per_epis": 2,
                    "use_lq_terminal_value": True,
                    "use_model_terminal_value": True,
                },
            )


class WelfareRolloutTest(unittest.TestCase):
    def test_matches_rbc_welfare_eval_and_uses_common_random_numbers(self):
        model, env, net, params, _ = _shared_setup()
        horizon, n_epis = 16, 8
        welfare_fn = create_welfare_fn(env, horizon, n_epis, init_range=0)
        policy = lambda obs: net.apply(params, obs)
        rollout = welfare_fn(policy, random.PRNGKey(3))
        self.assertEqual(rollout.states.shape, (n_epis, horizon, env.obs_dim))
        self.assertEqual(rollout.actions.shape, (n_epis, horizon, env.action_dim))
        self.assertEqual(rollout.welfare_per_episode.shape, (n_epis,))

        reference = create_welfare_eval_fn(
            model, policy_fn=lambda p, obs: net.apply(p, obs), horizon=horizon, n_epis=n_epis, init_range=0
        )(params, random.PRNGKey(3))
        np.testing.assert_allclose(rollout.welfare, reference.welfare, rtol=1e-5)

        again = welfare_fn(policy, random.PRNGKey(3))
        np.testing.assert_array_equal(rollout.states, again.states)
        no_shocks = create_welfare_fn(env, horizon, n_epis, init_range=0, simul_vol_scale=0.0)
        steady = no_shocks(lambda obs: env.deterministic_steady_state_action(), random.PRNGKey(3))
        np.testing.assert_allclose(steady.welfare, env.deterministic_steady_state_welfare(horizon), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
