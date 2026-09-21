#!/usr/bin/env python3
"""Train the core analytical policy-gradient algorithm on the time-to-build economy.

This trainer uses only the ``WelfareEnvironment`` contract, so it has no
Euler or RBC-specific welfare diagnostics; ``run_experiment`` records the
rollout loss, and ``APG.algorithm.create_welfare_fn`` evaluates the trained
policy afterwards. ``train(config)`` returns the ``run_experiment`` result
together with the environment, the network, and the log-linear solution, so
that experiment scripts can compare variants.

Config keys beyond ``run_experiment``'s:

    env                 keyword arguments for ``TimeToBuildRbc``
    loglinear_baseline  residual network around the LQ policy
    loglinear_normalize scale network inputs and outputs by the LQ stationary
                        standard deviations, with or without the baseline
"""

import os
import sys

import jax.numpy as jnp
from jax import config as jax_config

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from APG.algorithm import create_epoch_train_fn, create_eval_fn
from APG.environments import TimeToBuildRbc
from APG.loglinear import build_actor, print_loglinear_solution, solve_loglinear
from APG.training import run_experiment
from APG.training.plots import plot_learning_rate_schedule, plot_training_metrics
from DEQN.econ_models.RBC.train_shared import with_derived_counts

TTB_TRAIN = {
    "seed": 42,
    "run_name": "ttb_kp_lq_baseline",
    "date": "Sep2026",
    "env": {},
    "layers": [16, 16],
    "double_precision": False,
    "learning_rate": 0.025,
    "cosine_alpha": 0.01,
    "max_grad_norm": None,
    "n_epochs": 20,
    "steps_per_epoch": 5,
    "epis_per_step": 16,
    "periods_per_epis": 512,
    "antithetic_episodes": True,
    "init_range": 5,
    "simul_vol_scale": 1.0,
    "batch_size": None,
    "eval_n_epis": 32,
    "eval_periods_per_epis": 256,
    "use_terminal_value": False,
    "use_model_terminal_value": False,
    "terminal_value_horizon": 512,
    "gae_lambda": 0.95,
    "rematerialize_rollout": False,
    "loglinear_baseline": True,
    "loglinear_normalize": True,
    "loglinear_solver": "scipy",
    "checkpoint_every_n_epochs": 10,
    "save_orbax_checkpoint": False,
    "generate_plots": False,
    "working_dir": os.path.join(repo_root, "APG", "results"),
}


def build_env(config):
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    return TimeToBuildRbc(
        **config.get("env", {}),
        double_precision=config["double_precision"],
        precision=precision,
    )


def prepare_scales_and_baseline(env, config, verbose=True):
    """Solve the LQ policy when the baseline or the normalization asks for it.

    Returns the solution to hand to ``build_actor`` (``None`` when the network
    should not carry the linear baseline) and the raw solution, if any.
    """
    use_baseline = bool(config.get("loglinear_baseline", False))
    normalize = bool(config.get("loglinear_normalize", False))
    if not (use_baseline or normalize):
        return None, None
    solution = solve_loglinear(env, solver=config.get("loglinear_solver", "scipy"))
    if verbose:
        print_loglinear_solution(solution)
    if normalize:
        env.set_scales(solution.states_sd, solution.policies_sd)
    config["loglinear_C"] = solution.C.tolist()
    config["loglinear_states_sd"] = solution.states_sd.tolist()
    config["loglinear_policies_sd"] = solution.policies_sd.tolist()
    config["loglinear_spectral_radius"] = solution.spectral_radius
    return (solution if use_baseline else None), solution


def train(config, env=None, verbose=True):
    """Run one training experiment; return ``run_experiment``'s result plus ``env``, ``neural_net``, ``loglinear``."""
    config = with_derived_counts(dict(config))
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)
    if config.get("use_terminal_value") and config.get("use_model_terminal_value"):
        raise ValueError("learned critic bootstrap and model terminal continuation cannot be combined")
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if env is None:
        env = build_env(config)
    baseline, loglinear = prepare_scales_and_baseline(env, config, verbose=verbose)
    neural_net = build_actor(env, config, precision, solution=baseline)

    result = run_experiment(
        config=config,
        env=env,
        neural_net=neural_net,
        epoch_train_fn=create_epoch_train_fn(env, config),
        eval_fn=create_eval_fn(env, config),
    )
    if result and config.get("generate_plots", False):
        plots_dir = os.path.join(config["working_dir"], config["run_name"])
        plot_training_metrics(result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100)
        plot_learning_rate_schedule(result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100)
    result["env"] = env
    result["neural_net"] = neural_net
    result["loglinear"] = loglinear
    return result


def main():
    print(f"Training: {TTB_TRAIN['run_name']}", flush=True)
    return train(TTB_TRAIN)


if __name__ == "__main__":
    main()
