"""Exact-kink APG trainer: beta schedule and one training step."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax import random

from APG.constrained.exact_kink import create_exact_kink_epoch_train_fn, log_linear_beta
from APG.environments import RbcMultiSector
from APG.neural_nets import PolicyNet


class ExactKinkTrainerTest(unittest.TestCase):
    def test_beta_anneal_is_log_linear(self):
        start = log_linear_beta(jnp.array(0), 101, 10.0, 500.0)
        mid = log_linear_beta(jnp.array(50), 101, 10.0, 500.0)
        end = log_linear_beta(jnp.array(100), 101, 10.0, 500.0)
        np.testing.assert_allclose(start, 10.0, rtol=1e-5)
        np.testing.assert_allclose(end, 500.0, rtol=1e-5)
        np.testing.assert_allclose(mid, jnp.sqrt(10.0 * 500.0), rtol=1e-4)

    def test_one_epoch_runs_and_updates_parameters(self):
        env = RbcMultiSector(N=1, phi=0.0, i_min_frac=0.8, policy_map="exact_kink", project_investment=False)
        net = PolicyNet(features=[4], n_out=1, output_bias_init=1.0)
        params = net.init(random.PRNGKey(0), env.obs_ss)
        ts = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=optax.sgd(1e-3))
        train_fn = create_exact_kink_epoch_train_fn(
            env,
            net.apply,
            {
                "n_epochs": 1,
                "steps_per_epoch": 1,
                "epis_per_step": 1,
                "periods_per_epis": 4,
                "vk_horizon": 4,
                "init_range": 0,
                "simul_vol_scale": 0.0,
                "antithetic_episodes": False,
                "use_terminal_value": False,
                "n_euler_states": 8,
                "euler_grid_size": 4,
                "policy_beta_start": 10.0,
                "policy_beta_end": 20.0,
                "euler_weight": 0.1,
            },
        )
        new_state, _, _ = train_fn(ts, random.PRNGKey(1))
        before = jax.tree_util.tree_leaves(ts.params)
        after = jax.tree_util.tree_leaves(new_state.params)
        self.assertTrue(any(not bool(jnp.allclose(a, b)) for a, b in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
