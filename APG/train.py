#!/usr/bin/env python3
"""Train the core analytical policy-gradient algorithm."""

import os
import sys

import jax.numpy as jnp
from jax import config as jax_config

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from APG.algorithm import create_epoch_train_fn, create_eval_fn
from APG.environments import RbcMultiSector
from APG.loglinear import apply_lq_design, build_actor, prepare_loglinear
from APG.training import run_experiment
from APG.training.plots import (
    plot_learning_rate_schedule,
    plot_training_metrics,
)
from DEQN.econ_models.RBC.euler_eval import create_euler_eval_fn
from DEQN.econ_models.RBC.train_shared import (
    APG_LEARNING_RATE,
    DEQN_EVAL_MC_DRAWS,
    SHARED_RBC_TRAIN,
    shared_model_kwargs,
    with_derived_counts,
)
from DEQN.econ_models.RBC.welfare_eval import create_welfare_eval_fn


config = with_derived_counts(
    {
        **SHARED_RBC_TRAIN,
        "run_name": "rbc_policy_baseline",
        "date": "Aug2026",
        "learning_rate": APG_LEARNING_RATE,
        "checkpoint_every_n_epochs": 10,
        "save_orbax_checkpoint": True,
        "generate_plots": True,
        "use_terminal_value": False,
        "use_model_terminal_value": False,
        "terminal_value_horizon": 512,
        "gae_lambda": 0.95,
        "algorithm_schema": "apg_v1",
        "layers_critic": SHARED_RBC_TRAIN["layers"],
        "antithetic_episodes": False,
        "rematerialize_rollout": False,
        "loglinear_baseline": False,
        "loglinear_normalize": True,
        "loglinear_solver": "scipy",
        "use_lq_terminal_value": False,
        "working_dir": os.path.join(repo_root, "APG", "results"),
    }
)


def main():
    print(f"Training: {config['run_name']}", flush=True)
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)
    if sum(
        bool(config.get(key, False))
        for key in ("use_terminal_value", "use_model_terminal_value", "use_lq_terminal_value")
    ) > 1:
        raise ValueError(
            "learned critic, model terminal continuation, and LQ terminal value "
            "cannot be combined"
        )

    precision = jnp.float64 if config["double_precision"] else jnp.float32
    model_kwargs = shared_model_kwargs(config)
    n_sectors = model_kwargs.pop("n_sectors")
    env = RbcMultiSector(
        N=n_sectors,
        **model_kwargs,
        project_investment=True,
        policy_map="saving_rate",
        double_precision=config["double_precision"],
        precision=precision,
    )

    from APG.loglinear.design import needs_lq_objects

    if needs_lq_objects(config) or config.get("design") == "lq":
        loglinear = apply_lq_design(env, env.econ, config)
    else:
        loglinear = prepare_loglinear(env, config)
    actor_solution = loglinear if config.get("loglinear_baseline") else None
    neural_net = build_actor(env, config, precision, solution=actor_solution)

    def policy_fn(params, obs):
        output = neural_net.apply(params, obs)
        return output[0] if config.get("use_terminal_value") else output

    welfare_eval_fn = create_welfare_eval_fn(
        env.econ,
        policy_fn=policy_fn,
        horizon=config.get("welfare_history_horizon", config["welfare_horizon"]),
        n_epis=config["welfare_n_epis"],
        init_range=config["welfare_init_range"],
        init_range_a=config.get("welfare_init_range_a"),
    )
    euler_eval_fn = create_euler_eval_fn(
        env.econ,
        policy_fn=policy_fn,
        periods_per_epis=config["eval_periods_per_epis"],
        n_epis=config["eval_n_epis"],
        mc_draws=DEQN_EVAL_MC_DRAWS,
        init_range=config["init_range"],
        init_range_a=config.get("init_range_a"),
        simul_vol_scale=config["simul_vol_scale"],
    )
    result = run_experiment(
        config=config,
        env=env,
        neural_net=neural_net,
        epoch_train_fn=create_epoch_train_fn(env, config),
        eval_fn=create_eval_fn(env, config),
        welfare_eval_fn=welfare_eval_fn,
        euler_eval_fn=euler_eval_fn,
    )

    if result and config.get("generate_plots", True):
        plots_dir = os.path.join(config["working_dir"], config["run_name"])
        plot_training_metrics(
            training_results=result,
            save_dir=plots_dir,
            experiment_name=config["run_name"],
            display_dpi=100,
        )
        plot_learning_rate_schedule(
            training_results=result,
            save_dir=plots_dir,
            experiment_name=config["run_name"],
            display_dpi=100,
        )
    return result


if __name__ == "__main__":
    main()
