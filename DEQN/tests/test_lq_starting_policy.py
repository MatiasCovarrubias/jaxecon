"""DEQN and APG residual actors implement the same LQ saving-rate map."""

import unittest

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from jax import random

from APG.environments import RbcMultiSector
from APG.loglinear.design import apply_lq_design, lq_design
from APG.neural_nets import PolicyNetLoglinear
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet as NeuralNet_loglinear


def _env():
    return RbcMultiSector(
        N=1,
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        rho=0.9,
        shock_sd=0.04,
        phi=2.0,
        eps_c=0.5,
        double_precision=True,
        precision=jnp.float64,
    )


def _last_dense_is_zero(params):
    leaves = flatten_dict(params)
    last_kernel = None
    last_bias = None
    for path, value in leaves.items():
        name = "/".join(str(part) for part in path)
        if name.endswith("kernel"):
            last_kernel = value
        if name.endswith("bias"):
            last_bias = value
    return (
        last_kernel is not None
        and last_bias is not None
        and bool(jnp.all(last_kernel == 0))
        and bool(jnp.all(last_bias == 0))
    )


class LqStartingPolicyTest(unittest.TestCase):
    def test_deqn_and_apg_saving_rates_match_at_init(self):
        env = _env()
        design = lq_design(env)
        config = {}
        apply_lq_design(env, env.econ, config, design=design)

        features = [8, 8]
        deqn_net = NeuralNet_loglinear(
            features=features,
            C=jnp.asarray(design.C),
            states_sd=jnp.asarray(env.state_sd),
            policies_sd=jnp.asarray(env.policies_sd),
            param_dtype=jnp.float64,
        )
        apg_net = PolicyNetLoglinear(
            features=features,
            n_out=env.action_dim,
            C=jnp.asarray(design.C),
            states_sd=jnp.asarray(env.state_sd),
            policies_sd=jnp.asarray(env.policies_sd),
            precision=jnp.float64,
        )
        key = random.PRNGKey(0)
        dummy = jnp.zeros_like(env.obs_ss)
        deqn_params = deqn_net.init(key, dummy)
        apg_params = apg_net.init(key, dummy)
        self.assertTrue(_last_dense_is_zero(deqn_params))
        self.assertTrue(_last_dense_is_zero(apg_params))

        states = random.normal(random.PRNGKey(1), shape=(16, env.obs_dim), dtype=jnp.float64)
        for state in states:
            deqn_action = deqn_net.apply(deqn_params, state)
            apg_action = apg_net.apply(apg_params, state)
            capital, productivity = env.econ._capital_and_productivity(state)
            output = env.econ.production(capital, productivity)
            deqn_s = env.econ.saving_rate_from_policy(deqn_action, output)
            apg_s = env.econ.saving_rate_from_policy(apg_action, output)
            np.testing.assert_allclose(deqn_s, apg_s, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(deqn_action, apg_action, atol=1e-5, rtol=1e-5)
