"""Shared RBC training setup for DEQN and APG.

The public RBC trainers import this so a comparison run uses the same model,
PolicyNet width, shock simulation budget, Adam + cosine optimizer, and
parameter initialization. Algorithm-specific pieces stay in each trainer:
DEQN's Euler or KKT residual (including MC draws) and APG's differentiable return.

Both public trainers use the same Adam learning rate (`SHARED_LEARNING_RATE`).
The comparison grid is five halvings with largest value 0.05.

Both public trainers take `split(PRNGKey(seed), 5)` so `init`, `epoch`,
`eval`, and `welfare` match. APG peels its Euler-eval key off the leftover
stream. DEQN Monte Carlo draws are not a top-level stream; they are split
from each training step key after the occupancy shocks have been drawn.

`batch_size=None` means one Adam step over the whole simulated step
(`n_batches=1`). DEQN still flattens and shuffles periods across episodes;
that shuffle does not split the update when there is a single batch.

`antithetic_episodes` pairs each independent shock path with its negation,
from the same initial state. DEQN concatenates both occupancy paths. APG
averages the two discounted returns. `epis_per_step` counts independent
draws; the trajectory count doubles when the flag is on.

To make the SGD step even closer, turn off that shuffle in
`DEQN/algorithm/epoch_train.py` (`random.permutation` over flattened periods).
APG does not shuffle. The permutation is left on because that file is
production DEQN (RbcProdNet).

Optional portable snapshots (``policy_snapshot_every_n_epochs`` and/or
``policy_snapshot_epochs``) write ``params_epoch_XXXX.msgpack`` next to the
final ``params.msgpack``. Both keys default off.
"""

import math
import os

import optax
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from jax import numpy as jnp
from jax import random

SHARED_RBC_TRAIN = {
    "seed": 42,
    "n_sectors": 1,
    "beta": 0.99,
    "alpha": 0.3,
    "delta": 0.05,
    "shock_sd": 0.02,
    "rho": 0.7,
    "phi": 2.0,
    "i_min_frac": 0.0,
    "eps_c": 0.5,
    "cbar_frac": 0.0,
    "layers": [16, 16],
    "double_precision": False,
    "n_epochs": 20,
    "steps_per_epoch": 5,
    "epis_per_step": 16,
    "periods_per_epis": 512,
    "antithetic_episodes": True,
    "init_range": 5,
    "init_range_a": None,
    "simul_vol_scale": 1.0,
    "state_sampling": {"mode": "occupancy"},
    "batch_size": None,
    "cosine_alpha": 0.01,
    "max_grad_norm": None,
    "eval_n_epis": 32,
    "eval_periods_per_epis": 256,
    "welfare_horizon": 1024,
    "welfare_history_horizon": 256,
    "welfare_n_epis": 1024,
    "welfare_init_range": 0,
    "welfare_init_range_a": None,
    "welfare_every_n_epochs": 1,
    "post_training_analysis": True,
    "deqn_stop_gradient": True,
    "deqn_update": "residual",
}

SHARED_LEARNING_RATE = 0.025
DEQN_LEARNING_RATE = SHARED_LEARNING_RATE
APG_LEARNING_RATE = SHARED_LEARNING_RATE
DEQN_MC_DRAWS = 32
DEQN_EVAL_MC_DRAWS = 32
SHARED_MODEL_KEYS = (
    "n_sectors",
    "beta",
    "alpha",
    "delta",
    "shock_sd",
    "rho",
    "phi",
    "i_min_frac",
    "eps_c",
    "cbar_frac",
)
IRREVERSIBLE_RBC_OVERLAY = {
    "i_min_frac": 0.975,
    "phi": 0.0,
}


def split_shared_rbc_rngs(seed, extra_names=()):
    """Return the shared RBC streams, plus any trainer-only leftovers.

    ``init``, ``epoch``, ``eval``, and ``welfare`` are the 5-way split both
    trainers must share. Names in ``extra_names`` (APG's ``euler``) are
    split sequentially from the leftover key so they cannot desynchronize
    the training shocks.
    """
    leftover, rng_init, rng_epoch, rng_eval, rng_welfare = random.split(
        random.PRNGKey(int(seed)), 5
    )
    streams = {
        "leftover": leftover,
        "init": rng_init,
        "epoch": rng_epoch,
        "eval": rng_eval,
        "welfare": rng_welfare,
    }
    for name in extra_names:
        leftover, extra = random.split(leftover)
        streams["leftover"] = leftover
        streams[name] = extra
    return streams


def shared_model_kwargs(config=None):
    cfg = SHARED_RBC_TRAIN if config is None else config
    return {key: cfg.get(key, SHARED_RBC_TRAIN[key]) for key in SHARED_MODEL_KEYS}


def with_irreversible(config, enabled=False):
    if not enabled:
        return config
    config = dict(config)
    config.update(IRREVERSIBLE_RBC_OVERLAY)
    if "exper_name" in config:
        config["exper_name"] = "rbc_irreversible"
    if "run_name" in config:
        config["run_name"] = "rbc_policy_irreversible"
    return config


def with_rbc_state_sampling(
    config,
    mode="occupancy",
    *,
    grid_share=0.5,
    k_rel_min=0.90,
    k_rel_max=1.16,
    a_sd_min=-2.5,
    a_sd_max=2.5,
    n_k=33,
    n_a=33,
):
    """Configure DEQN state draws for the one-sector RBC."""
    if mode not in ("occupancy", "grid", "mixed"):
        raise ValueError("mode must be occupancy, grid, or mixed")
    config = dict(config)
    if mode == "occupancy":
        config["state_sampling"] = {"mode": mode}
        return config
    if int(config["n_sectors"]) != 1:
        raise ValueError("the RBC grid preset currently supports one sector")
    if not 0 < k_rel_min < k_rel_max:
        raise ValueError("capital grid bounds must satisfy 0 < k_rel_min < k_rel_max")

    stationary_a_sd = float(config["shock_sd"]) / math.sqrt(1 - float(config["rho"]) ** 2)
    config["state_sampling"] = {
        "mode": mode,
        "grid_share": float(grid_share),
        "grid_min": [math.log(float(k_rel_min)), float(a_sd_min) * stationary_a_sd],
        "grid_max": [math.log(float(k_rel_max)), float(a_sd_max) * stationary_a_sd],
        "grid_points": [int(n_k), int(n_a)],
        "grid_include_zero": True,
        "economic_bounds": {
            "k_rel": [float(k_rel_min), float(k_rel_max)],
            "a_stationary_sd": [float(a_sd_min), float(a_sd_max)],
        },
    }
    return config


def lr_halves_grid(center, n=5, largest=0.05):
    """Geometric grid of `n` learning rates with ratio 1/2, ending at `largest`.

    `center` must lie on that grid. The default largest value is 0.05.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    grid = [largest * (0.5 ** (n - 1 - i)) for i in range(n)]
    if not any(abs(value - center) < 1e-12 for value in grid):
        raise ValueError(f"center={center} is not on the grid ending at {largest}")
    return grid


def trajectory_multiplier(config):
    return 2 if config.get("antithetic_episodes") else 1


def with_derived_counts(config):
    """Fill in ``n_trajectories_per_step``, ``periods_per_step``, ``batch_size``, ``n_batches``.

    Re-applying this after changing the rollout sizes is safe: a ``batch_size``
    that was previously derived as the whole step (``n_batches == 1``) is
    re-derived rather than kept.
    """
    config = dict(config)
    n_traj = config["epis_per_step"] * trajectory_multiplier(config)
    periods_per_step = config["periods_per_epis"] * n_traj
    batch_size = config.get("batch_size")
    previously_whole_step = (
        config.get("n_batches") == 1 and batch_size == config.get("periods_per_step")
    )
    if batch_size is None or previously_whole_step:
        batch_size = periods_per_step
    if periods_per_step % batch_size != 0:
        raise ValueError(f"periods_per_step={periods_per_step} must be divisible by batch_size={batch_size}")
    config["n_trajectories_per_step"] = n_traj
    config["periods_per_step"] = periods_per_step
    config["batch_size"] = batch_size
    config["n_batches"] = periods_per_step // batch_size
    return config


def create_optimizer(config):
    if callable(config["learning_rate"]):
        lr_schedule = config["learning_rate"]
    else:
        total_steps = max(1, config["n_epochs"] * config["steps_per_epoch"])
        lr_schedule = optax.cosine_decay_schedule(
            init_value=config["learning_rate"],
            decay_steps=total_steps,
            alpha=config.get("cosine_alpha", 0.01),
        )
    tx = optax.adam(lr_schedule)
    max_grad_norm = config.get("max_grad_norm")
    if max_grad_norm:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx, lr_schedule


def init_policy_params(neural_net, rng, state_ss):
    return neural_net.init(rng, jnp.zeros_like(state_ss))


def create_policy_train_state(neural_net, params, config):
    tx, lr_schedule = create_optimizer(config)
    train_state_obj = train_state.TrainState.create(
        apply_fn=neural_net.apply,
        params=params,
        tx=tx,
    )
    return train_state_obj, lr_schedule


def save_policy_params(params, path):
    with open(path, "wb") as handle:
        handle.write(to_bytes(params))


def load_policy_params(template, path):
    with open(path, "rb") as handle:
        return from_bytes(template, handle.read())


def _as_snapshot_epochs(value):
    if value is None or value is False:
        return ()
    if isinstance(value, bool):
        raise TypeError("policy_snapshot_epochs must be a sequence of ints")
    if isinstance(value, (int, float)):
        return (int(value),)
    if isinstance(value, (str, bytes)):
        raise TypeError("policy_snapshot_epochs must be a sequence of ints")
    return tuple(int(epoch) for epoch in value)


def resolve_policy_snapshot_epochs(config):
    """Epochs that should write a portable ``params_epoch_*.msgpack`` file.

    Off by default. Enable with ``policy_snapshot_every_n_epochs`` (positive
    int) and/or ``policy_snapshot_epochs`` (int or sequence). When enabled,
    epoch 0 and the final epoch are always included.
    """
    every = config.get("policy_snapshot_every_n_epochs")
    if isinstance(every, bool):
        raise TypeError("policy_snapshot_every_n_epochs must be a positive int")
    every_on = every is not None and int(every) > 0
    explicit = _as_snapshot_epochs(config.get("policy_snapshot_epochs"))
    if not every_on and not explicit:
        return frozenset()

    n_epochs = int(config["n_epochs"])
    epochs = set(explicit)
    if every_on:
        epochs.update(range(0, n_epochs + 1, int(every)))
    epochs.add(0)
    epochs.add(n_epochs)
    return frozenset(epoch for epoch in epochs if 0 <= epoch <= n_epochs)


def policy_snapshot_path(run_dir, epoch):
    return os.path.join(os.fspath(run_dir), f"params_epoch_{int(epoch):04d}.msgpack")


def maybe_save_policy_snapshot(params, run_dir, epoch, snapshot_epochs):
    epoch = int(epoch)
    if epoch not in snapshot_epochs:
        return None
    os.makedirs(os.fspath(run_dir), exist_ok=True)
    path = policy_snapshot_path(run_dir, epoch)
    save_policy_params(params, path)
    return path
