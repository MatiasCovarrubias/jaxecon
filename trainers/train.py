"""Dispatch a shared model to APG, DEQN, or time iteration."""

from econ_models.check import check_algorithm

from trainers.apg import train_apg
from trainers.deqn import train_deqn
from trainers.time_iteration import train_time_iteration

_TRAIN = {
    "apg": train_apg,
    "deqn": train_deqn,
    "time_iteration": train_time_iteration,
}


def default_config():
    """Short settings used by the smoke script."""
    return {
        "seed": 0,
        "learning_rate": 1e-2,
        "hidden": 8,
        "control_width": 0.05,
        "epochs": 1,
        "steps_per_epoch": 1,
        "episodes": 2,
        "periods": 4,
        "batch_size": 4,
        "mc_draws": 2,
        "n_a": 3,
        "n_k": 5,
        "ti_iterations": 2,
        "newton_steps": 4,
    }


def train(model, algorithm, config=None):
    """Check ``model`` for ``algorithm``, then run that trainer.

    ``config`` overrides :func:`default_config`. The result is a dict of floats.
    """
    if algorithm not in _TRAIN:
        raise ValueError(f"algorithm must be one of {tuple(_TRAIN)}")
    settings = default_config()
    if config:
        settings.update(config)
    check_algorithm(model, algorithm)
    return _TRAIN[algorithm](model, settings)
