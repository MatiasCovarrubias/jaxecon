"""Primal-dual (learned multiplier) APG: what the multiplier does and what it must not touch."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from flax.training import train_state

from APG.constrained.networks import MultiplierNet
from APG.constrained.primal_dual import (
    ConstrainedTrainState,
    create_constrained_epoch_train_fn,
    create_constrained_objectives,
)
from APG.environments import RbcMultiSector
from APG.neural_nets import PolicyNet


def _config(**overrides):
    return {
        "periods_per_epis": 3,
        "epis_per_step": 2,
        "steps_per_epoch": 1,
        "init_range": 0,
        "simul_vol_scale": 0.0,
        "antithetic_episodes": True,
        "augmented_lagrangian_rho": 1.0,
        "constraint_grid_share": 0.5,
        "constraint_grid_size": 5,
        **overrides,
    }


def _env(**overrides):
    kwargs = {"N": 1, "project_investment": False, "i_min_frac": 0.975, "phi": 0.0, **overrides}
    return RbcMultiSector(**kwargs)


def _networks_and_params(env):
    actor = PolicyNet(features=[4], n_out=1)
    multiplier = MultiplierNet(features=[4], initial_value=0.1)
    actor_key, multiplier_key = jax.random.split(jax.random.PRNGKey(0))
    return actor, multiplier, actor.init(actor_key, env.obs_ss), multiplier.init(multiplier_key, env.obs_ss)


def _violating_actor_params(actor_params):
    """Force the saving-rate logit far negative so investment sits below the floor."""
    params = unfreeze(actor_params)
    params["params"]["Dense_1"]["kernel"] = jnp.zeros_like(params["params"]["Dense_1"]["kernel"])
    params["params"]["Dense_1"]["bias"] = jnp.full_like(params["params"]["Dense_1"]["bias"], -10.0)
    return freeze(params)


def _train_state(actor, multiplier, actor_params, multiplier_params, multiplier_tx):
    return ConstrainedTrainState(
        actor=train_state.TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.set_to_zero()),
        multiplier=train_state.TrainState.create(apply_fn=multiplier.apply, params=multiplier_params, tx=multiplier_tx),
    )


class MultiplierNetworkTest(unittest.TestCase):
    def test_multiplier_is_positive_and_starts_at_the_requested_value(self):
        env = _env()
        _, multiplier, _, params = _networks_and_params(env)
        grid = env.constraint_state_grid(size=5)
        values = multiplier.apply(params, grid)
        self.assertTrue(bool(jnp.all(values > 0)))
        np.testing.assert_allclose(values, 0.1, rtol=1e-5, atol=1e-6)

    def test_constraint_grid_covers_the_requested_domain(self):
        env = _env()
        states = env.constraint_state_grid(size=7)
        capital, productivity = env.econ._capital_and_productivity(states)
        stationary_sd = env.shock_sd[0] / jnp.sqrt(1 - env.rho[0] ** 2)
        self.assertEqual(states.shape, (49, env.obs_dim))
        np.testing.assert_allclose(capital.min() / env.K_ss[0], 0.90, rtol=1e-6)
        np.testing.assert_allclose(capital.max() / env.K_ss[0], 1.16, rtol=1e-6)
        np.testing.assert_allclose(productivity.min() / stationary_sd, -2.5, rtol=1e-6)
        np.testing.assert_allclose(productivity.max() / stationary_sd, 2.5, rtol=1e-6)


class PrimalDualTest(unittest.TestCase):
    def test_actor_and_multiplier_objectives_are_detached_from_each_other(self):
        env = _env()
        actor, multiplier, actor_params, multiplier_params = _networks_and_params(env)
        actor_objective, multiplier_objective = create_constrained_objectives(env, actor.apply, multiplier.apply, _config())
        keys = jax.random.split(jax.random.PRNGKey(1), 2)
        actor_cross = jax.grad(lambda p: actor_objective(actor_params, p, keys)[0])(multiplier_params)
        multiplier_cross = jax.grad(lambda p: multiplier_objective(multiplier_params, p, keys)[0])(actor_params)
        for leaf in jax.tree_util.tree_leaves(actor_cross) + jax.tree_util.tree_leaves(multiplier_cross):
            np.testing.assert_allclose(leaf, 0, atol=1e-9)

    def test_dual_step_raises_multiplier_where_the_floor_is_violated(self):
        env = _env(shock_sd=0.0)
        actor, multiplier, actor_params, multiplier_params = _networks_and_params(env)
        state = _train_state(actor, multiplier, _violating_actor_params(actor_params), multiplier_params, optax.sgd(0.1))
        train_epoch = jax.jit(create_constrained_epoch_train_fn(env, actor.apply, multiplier.apply, _config()))

        before = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])
        state, _, metrics = train_epoch(state, jax.random.PRNGKey(2))
        after = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])

        self.assertGreater(after, before)
        self.assertGreater(float(metrics.objective.grid_violation_frac[0]), 0)

    def test_dual_step_lowers_multiplier_where_the_floor_is_slack(self):
        env = _env(shock_sd=0.0, i_min_frac=0.1)
        actor, multiplier, actor_params, multiplier_params = _networks_and_params(env)
        state = _train_state(actor, multiplier, actor_params, multiplier_params, optax.sgd(0.1))
        train_epoch = jax.jit(create_constrained_epoch_train_fn(env, actor.apply, multiplier.apply, _config()))

        before = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])
        state, _, _ = train_epoch(state, jax.random.PRNGKey(3))
        after = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])
        self.assertLess(after, before)

    def test_projected_dual_update_recovers_from_a_tiny_multiplier(self):
        env = _env(shock_sd=0.0)
        actor, multiplier, actor_params, multiplier_params = _networks_and_params(env)
        multiplier_params = unfreeze(multiplier_params)
        multiplier_params["params"]["Dense_1"]["bias"] = jnp.full_like(multiplier_params["params"]["Dense_1"]["bias"], -20.0)
        multiplier_params = freeze(multiplier_params)
        state = _train_state(actor, multiplier, _violating_actor_params(actor_params), multiplier_params, optax.adam(0.01))
        train_epoch = jax.jit(
            create_constrained_epoch_train_fn(
                env,
                actor.apply,
                multiplier.apply,
                _config(multiplier_update_mode="projected", multiplier_projected_step_size=0.01, multiplier_violation_weight=10.0),
            )
        )

        before = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])
        state, _, _ = train_epoch(state, jax.random.PRNGKey(4))
        after = float(multiplier.apply(state.multiplier.params, env.obs_ss)[0])
        self.assertGreater(after, before)

    def test_model_terminal_value_changes_the_reported_true_return(self):
        env = _env()
        actor, multiplier, actor_params, multiplier_params = _networks_and_params(env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        common = _config(constraint_grid_share=0.0, periods_per_epis=2)
        _, plain = create_constrained_objectives(env, actor.apply, multiplier.apply, common)[0](actor_params, multiplier_params, keys)
        _, tailed = create_constrained_objectives(
            env, actor.apply, multiplier.apply, {**common, "use_model_terminal_value": True, "terminal_value_horizon": 16}
        )[0](actor_params, multiplier_params, keys)
        self.assertLess(float(tailed.true_return), float(plain.true_return))


if __name__ == "__main__":
    unittest.main()
