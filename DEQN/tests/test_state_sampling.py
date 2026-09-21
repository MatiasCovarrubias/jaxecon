from types import SimpleNamespace

import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from DEQN.algorithm.simulation import (
    create_episode_simul_fn,
    create_normalized_state_grid,
    create_step_state_sampler,
    sample_balanced_grid,
)


class DummyModel:
    state_ss = jnp.zeros(2)

    @staticmethod
    def initial_state(_rng, _init_range, mode=None):
        return jnp.zeros(2)

    @staticmethod
    def sample_shock(_rng):
        return jnp.zeros(1)

    @staticmethod
    def step(state, _policy, _shock):
        return state + jnp.array([10.0, 0.0])


def config(mode="occupancy", grid_share=0.5):
    return {
        "epis_per_step": 2,
        "periods_per_epis": 3,
        "antithetic_episodes": True,
        "init_range": 0,
        "simul_vol_scale": 1.0,
        "periods_per_step": 12,
        "state_sampling": {
            "mode": mode,
            "grid_share": grid_share,
            "grid_min": [-1.0, -2.0],
            "grid_max": [1.0, 2.0],
            "grid_points": [3, 3],
            "grid_include_zero": True,
        },
    }


def train_state():
    return SimpleNamespace(
        params=jnp.zeros(1),
        apply_fn=lambda _params, _state: jnp.zeros(1),
    )


def test_normalized_grid_has_bounds_and_steady_state():
    grid = np.asarray(create_normalized_state_grid(DummyModel(), config("grid")))

    assert grid.shape == (9, 2)
    np.testing.assert_allclose(grid.min(axis=0), [-1.0, -2.0])
    np.testing.assert_allclose(grid.max(axis=0), [1.0, 2.0])
    assert np.any(np.all(grid == 0, axis=1))


def test_balanced_grid_sampling_repeats_each_point_equally():
    grid = create_normalized_state_grid(DummyModel(), config("grid"))
    sampled = np.asarray(sample_balanced_grid(grid, 18, random.PRNGKey(0)))
    _, counts = np.unique(sampled, axis=0, return_counts=True)

    np.testing.assert_array_equal(counts, np.full(9, 2))


def test_occupancy_mode_matches_episode_simulation():
    cfg = config("occupancy")
    rng = random.PRNGKey(1)
    state = train_state()
    sampler = create_step_state_sampler(DummyModel(), cfg)
    sampled = sampler(state, rng)

    episode_fn = create_episode_simul_fn(DummyModel(), cfg)
    episode_rngs = random.split(rng, cfg["epis_per_step"])
    expected = jax.vmap(episode_fn, in_axes=(None, 0))(state, episode_rngs).reshape(12, 2)

    np.testing.assert_allclose(sampled, expected)


def test_mixed_mode_uses_configured_grid_share():
    cfg = config("mixed", grid_share=0.5)
    sampled = np.asarray(create_step_state_sampler(DummyModel(), cfg)(train_state(), random.PRNGKey(2)))
    grid_rows = np.all(np.abs(sampled) <= np.array([1.0, 2.0]), axis=1)

    assert sampled.shape == (12, 2)
    assert int(grid_rows.sum()) == 6
