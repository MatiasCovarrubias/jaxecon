"""Dispatch a shared model to APG, DEQN, or time iteration."""

from typing import Callable, NamedTuple

from econ_models.check import check_algorithm

from trainers.apg import train_apg
from trainers.deqn import train_deqn
from trainers.time_iteration import train_time_iteration

_TRAIN = {
    "apg": train_apg,
    "deqn": train_deqn,
    "time_iteration": train_time_iteration,
}


class Solution(NamedTuple):
    """Trained policy and the scalars reported by its trainer.

    ``policy(state)`` maps one state to one control in the model's coordinates.
    ``arrays`` is the savable object behind that policy: network layers for
    APG and DEQN, or the time-iteration grid.
    """

    metrics: dict
    policy: Callable
    arrays: object


def default_config():
    """One cheap Adam step, or two Newton sweeps. Tests use this."""
    return {
        "seed": 0,
        "learning_rate": 1e-2,
        "cosine_alpha": 0.01,
        "hidden": (8, 8),
        "epochs": 1,
        "steps_per_epoch": 1,
        "episodes": 2,
        "periods": 4,
        "tail_periods": 4,
        "antithetic": True,
        "n_a": 3,
        "n_k": 5,
        "k_min_rel": 0.4,
        "k_max_rel": 1.6,
        "ti_iterations": 2,
        "ti_tol": 1e-10,
        "newton_steps": 4,
    }


def experiment_config():
    """Locked comparison: 1000 Adam steps, 1000 time-iteration sweeps, 32 x 512.

    The shared model is float64. APG adds a 128-period steady-saving tail
    under zero shocks.
    """
    return {
        "seed": 0,
        "learning_rate": 0.025,
        "cosine_alpha": 0.01,
        "hidden": (16, 16),
        "epochs": 200,
        "steps_per_epoch": 5,
        "episodes": 32,
        "periods": 512,
        "tail_periods": 128,
        "antithetic": True,
        "n_a": 15,
        "n_k": 81,
        "k_min_rel": 0.4,
        "k_max_rel": 1.6,
        "ti_iterations": 1000,
        "ti_tol": 1e-10,
        "newton_steps": 15,
    }


def train(model, algorithm, config=None):
    """Check ``model`` for ``algorithm``, then run that trainer.

    ``config`` overrides :func:`default_config`. The comparison run is
    :func:`experiment_config`.
    """
    if algorithm not in _TRAIN:
        raise ValueError(f"algorithm must be one of {tuple(_TRAIN)}")
    settings = default_config()
    if config:
        settings.update(config)
    check_algorithm(model, algorithm)
    metrics, policy, arrays = _TRAIN[algorithm](model, settings)
    metrics["wall_seconds"] = float(metrics["run_seconds"])
    return Solution(metrics, policy, arrays)
