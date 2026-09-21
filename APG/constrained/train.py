#!/usr/bin/env python3
"""Train APG with an explicit investment-irreversibility constraint.

Run from the repository root:

    python -m APG.constrained.train                 # smooth RBC, projected investment
    python -m APG.constrained.train --irreversible  # i_min_frac=0.975, phi=0

``investment_constraint_mode`` in ``config`` selects the treatment of the
constraint: ``project`` (clip investment inside the model), ``penalty``
(utility-scaled shortfall penalty), ``learned_multiplier`` (primal-dual with a
multiplier network), or ``exact_kink`` (explicit kink policy map).
"""

import math
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
base_dir = os.path.join(repo_root, "APG")

import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402

from APG.algorithm import create_epoch_train_fn, create_eval_fn  # noqa: E402
from APG.constrained import (  # noqa: E402
    create_constrained_epoch_train_fn,
    create_constrained_eval_fn,
    create_exact_kink_epoch_train_fn,
    create_exact_kink_eval_fn,
    create_multiplier_warmup_fn,
    GridMultiplierNet,
    MultiplierNet,
    run_constrained_experiment,
)
from APG.environments import RbcMultiSector  # noqa: E402
from APG.neural_nets import (  # noqa: E402
    ActorCritic,
    PolicyNet,
)
from APG.training import run_experiment  # noqa: E402
from APG.training.plots import (  # noqa: E402
    plot_learning_rate_schedule,
    plot_training_metrics,
)
from DEQN.econ_models.RBC.euler_eval import create_euler_eval_fn  # noqa: E402
from DEQN.econ_models.RBC.plots import (  # noqa: E402
    collect_eval_histories,
    plot_eval_comparison,
)
from DEQN.econ_models.RBC.train_shared import (  # noqa: E402
    APG_LEARNING_RATE,
    DEQN_EVAL_MC_DRAWS,
    SHARED_RBC_TRAIN,
    shared_model_kwargs,
    with_derived_counts,
    with_irreversible,
)
from DEQN.econ_models.RBC.welfare_eval import create_welfare_eval_fn  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================


config = with_irreversible(
    with_derived_counts(
        {
            **SHARED_RBC_TRAIN,
            "run_name": "rbc_policy_baseline",
            "date": "Aug2026",
            "learning_rate": APG_LEARNING_RATE,
            "checkpoint_every_n_epochs": 10,
            "save_orbax_checkpoint": True,
            "generate_plots": True,
            "use_terminal_value": False,
            "gae_lambda": 0.95,
            "investment_constraint_mode": "project",
            "investment_penalty": 0.0,
            "multiplier_learning_rate": 0.001,
            "multiplier_architecture": "mlp",
            "multiplier_layers": SHARED_RBC_TRAIN["layers"],
            "multiplier_initial_value": 0.1,
            "multiplier_gradient_floor": 0.0,
            "multiplier_max_grad_norm": 1.0,
            "multiplier_violation_weight": 1.0,
            "multiplier_update_mode": "lagrangian",
            "multiplier_projected_step_size": 0.01,
            "multiplier_direct_relaxation": 1.0,
            "multiplier_direct_relaxation_floor": 0.01,
            "multiplier_direct_relaxation_decay": 1.0,
            "multiplier_predictor_corrector": False,
            "multiplier_min_value": 1e-8,
            "multiplier_warmup_steps": 0,
            "actor_steps_per_multiplier_update": 1,
            "alternate_actor_multiplier_updates": False,
            "augmented_lagrangian_rho": 1.0,
            "augmented_lagrangian_form": "hinge_quadratic",
            "constraint_grid_share": 0.5,
            "constraint_grid_size": 33,
            "constraint_grid_k_min": 0.90,
            "constraint_grid_k_max": 1.16,
            "constraint_grid_a_sd_min": -2.5,
            "constraint_grid_a_sd_max": 2.5,
            "select_best_checkpoint": False,
            "selection_max_occupancy_violation_frac": 0.02,
            "selection_max_grid_violation_frac": 0.01,
            "selection_violation_penalty": 10.0,
            "selection_euler_weight": 0.25,
            "algorithm_schema": "apg_v1",
            "layers_critic": SHARED_RBC_TRAIN["layers"],
            "policy_map": "saving_rate",
            "policy_beta_start": 10.0,
            "policy_beta_end": 500.0,
            "kappa_mu": None,
            "state_k_scale": 0.1,
            "investment_cap_frac": 0.9,
            "euler_weight": 1.0,
            "critic_coef": 1.0,
            "latent_reg_weight": 1e-4,
            "latent_reg_threshold": 5.0,
            "n_euler_states": 256,
            "euler_grid_share": 0.5,
            "euler_occupancy_share": 0.2,
            "euler_corner_share": 0.3,
            "euler_corner_band": 0.5,
            "euler_grid_size": 64,
            "vk_horizon": 48,
            "use_model_terminal_value": False,
            "terminal_value_horizon": 512,
            "rematerialize_rollout": False,
            "working_dir": os.path.join(base_dir, "results/"),
        }
    ),
    enabled="--irreversible" in sys.argv,
)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    print(f"Training: {config['run_name']}", flush=True)

    # Precision setup
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Create environment
    print("Creating environment...", flush=True)
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    model_kwargs = shared_model_kwargs(config)
    n_sectors = model_kwargs.pop("n_sectors")
    constraint_mode = config.get("investment_constraint_mode", "project")
    if constraint_mode not in ("project", "penalty", "learned_multiplier", "exact_kink"):
        raise ValueError(
            "investment_constraint_mode must be 'project', 'penalty', "
            "'learned_multiplier', or 'exact_kink'"
        )
    investment_penalty = float(config.get("investment_penalty", 0.0))
    if constraint_mode == "penalty" and investment_penalty <= 0:
        raise ValueError("penalty mode requires a positive investment_penalty")
    if constraint_mode == "project" and investment_penalty != 0:
        raise ValueError("project mode requires investment_penalty=0")
    if constraint_mode == "learned_multiplier":
        if investment_penalty != 0:
            raise ValueError("learned_multiplier mode requires investment_penalty=0")
        if n_sectors != 1:
            raise ValueError("learned_multiplier mode currently supports one sector")
        if not config.get("i_min_frac", 0):
            raise ValueError("learned_multiplier mode requires irreversible investment")
        if config["use_terminal_value"]:
            raise ValueError("learned_multiplier mode does not yet support a value head")
        if float(config.get("phi", 0)) != 0:
            raise ValueError("learned_multiplier mode currently requires phi=0")
        if config.get("use_model_terminal_value") and config.get("use_terminal_value"):
            raise ValueError(
                "model terminal continuation and a learned critic cannot be combined"
            )
        if config.get("multiplier_update_mode") == "phr_direct":
            config["algorithm_schema"] = "constrained_apg_learned_multiplier_v4"
        elif (
            config.get("multiplier_update_mode") == "projected"
            or config.get("augmented_lagrangian_form") == "phr"
            or config.get("multiplier_architecture") == "grid"
        ):
            config["algorithm_schema"] = "constrained_apg_learned_multiplier_v2"
        else:
            config["algorithm_schema"] = "constrained_apg_learned_multiplier_v1"
    if constraint_mode == "exact_kink":
        if n_sectors != 1:
            raise ValueError("exact_kink mode currently supports one sector")
        if float(config.get("phi", 0)) != 0:
            raise ValueError("exact_kink mode currently requires phi=0")
        if config.get("use_model_terminal_value") and config.get("use_terminal_value"):
            raise ValueError(
                "model terminal continuation and a learned critic cannot be combined"
            )
        config["policy_map"] = "exact_kink"
        if float(config.get("i_min_frac", 0)) > 0:
            config["algorithm_schema"] = "exact_kink_apg_v1"
        else:
            config["algorithm_schema"] = "unit_excess_apg_v1"
        if config.get("learning_rate") == APG_LEARNING_RATE:
            config["learning_rate"] = 1e-3
    env = RbcMultiSector(
        N=n_sectors,
        **model_kwargs,
        investment_penalty=investment_penalty,
        project_investment=constraint_mode == "project",
        policy_map=config.get("policy_map", "saving_rate"),
        policy_beta=config.get("policy_beta_start", 10.0),
        hard_floor=False,
        kappa_mu=config.get("kappa_mu"),
        state_k_scale=config.get("state_k_scale", 0.1),
        investment_cap_frac=config.get("investment_cap_frac", 0.9),
        double_precision=config["double_precision"],
        precision=precision,
    )
    print(f"  n_sectors: {config['n_sectors']}", flush=True)
    print(f"  beta: {float(env.beta):.2f}", flush=True)
    print(f"  alpha: {float(env.alpha[0]):.2f}", flush=True)
    print(f"  delta: {float(env.delta[0]):.2f}", flush=True)
    print(f"  shock_sd: {float(env.shock_sd[0]):.2f}", flush=True)
    print(f"  rho: {float(env.rho[0]):.2f}", flush=True)
    print(f"  phi: {float(env.phi):.2f}", flush=True)
    print(f"  i_min_frac: {float(env.i_min_frac):.3f}", flush=True)
    print(f"  investment constraint: {constraint_mode}", flush=True)
    print(f"  policy_map: {env.econ.policy_map}", flush=True)
    if constraint_mode == "exact_kink":
        print(
            "  exact-kink: "
            f"beta={config['policy_beta_start']:.3g}->{config['policy_beta_end']:.3g}, "
            f"kappa_I={float(env.econ.kappa_I[0]):.4g}, "
            f"kappa_mu={float(env.econ.kappa_mu):.4g}, "
            f"euler_weight={config['euler_weight']:.3g}, "
            f"vk_horizon={config['vk_horizon']}, "
            f"critic={config['use_terminal_value']}, "
            f"model_terminal={config.get('use_model_terminal_value', False)}",
            flush=True,
        )
    if constraint_mode == "penalty":
        print(f"  investment penalty: {investment_penalty:.4g}", flush=True)
    elif constraint_mode == "learned_multiplier":
        multiplier_optimizer = (
            "direct"
            if config["multiplier_update_mode"] == "phr_direct"
            else f"Adam lr={config['multiplier_learning_rate']:.4g}"
        )
        print(
            "  multiplier: "
            f"optimizer={multiplier_optimizer}, "
            f"rho={config['augmented_lagrangian_rho']:.4g}, "
            f"form={config['augmented_lagrangian_form']}, "
            f"grid_share={config['constraint_grid_share']:.2f}, "
            f"update={config['multiplier_update_mode']}, "
            f"actor_steps={config['actor_steps_per_multiplier_update']}, "
            f"dual_relax={config['multiplier_direct_relaxation']:.3g}"
            f"->{config['multiplier_direct_relaxation_floor']:.3g}, "
            f"dual_decay={config['multiplier_direct_relaxation_decay']:.5g}, "
            f"dual_predictor={config['multiplier_predictor_corrector']}, "
            f"gradient_floor={config['multiplier_gradient_floor']:.3g}, "
            f"violation_weight={config['multiplier_violation_weight']:.3g}, "
            f"model_terminal={config.get('use_model_terminal_value', False)}",
            flush=True,
        )
    print(f"  eps_c (IES): {float(env.econ.eps_c):.2f}", flush=True)
    print(f"  risk aversion: {float(env.econ.risk_aversion):.2f}", flush=True)
    print(f"  obs_dim: {env.obs_dim}", flush=True)
    print(f"  action_dim: {env.action_dim}", flush=True)

    # Create neural network
    print("Creating neural network...", flush=True)
    layers = config.get("layers", config.get("layers_actor", [32, 16]))
    if constraint_mode == "learned_multiplier":
        neural_net = PolicyNet(
            features=layers,
            n_out=env.action_dim,
            precision=precision,
        )
        multiplier_architecture = config.get("multiplier_architecture", "mlp")
        if multiplier_architecture == "grid":
            stationary_a_sd = float(env.shock_sd[0] / jnp.sqrt(1 - env.rho[0] ** 2))
            multiplier_net = GridMultiplierNet(
                grid_size=int(config.get("constraint_grid_size", 33)),
                state_min=(
                    math.log(float(config.get("constraint_grid_k_min", 0.90))),
                    stationary_a_sd * float(config.get("constraint_grid_a_sd_min", -2.5)),
                ),
                state_max=(
                    math.log(float(config.get("constraint_grid_k_max", 1.16))),
                    stationary_a_sd * float(config.get("constraint_grid_a_sd_max", 2.5)),
                ),
                initial_value=float(config.get("multiplier_initial_value", 0.1)),
                gradient_floor=float(config.get("multiplier_gradient_floor", 0.0)),
                precision=precision,
            )
        elif multiplier_architecture in ("mlp", "linear"):
            multiplier_net = MultiplierNet(
                features=config.get("multiplier_layers", layers),
                initial_value=float(config.get("multiplier_initial_value", 0.1)),
                gradient_floor=float(config.get("multiplier_gradient_floor", 0.0)),
                precision=precision,
            )
        else:
            raise ValueError("multiplier_architecture must be 'mlp', 'linear', or 'grid'")
    elif constraint_mode == "exact_kink" and config["use_terminal_value"]:
        neural_net = ActorCritic(
            actions_dim=env.action_dim,
            hidden_dims_actor=layers,
            hidden_dims_critic=config["layers_critic"],
            precision=precision,
            policy_output_bias_init=1.0,
        )
    elif constraint_mode == "exact_kink":
        neural_net = PolicyNet(
            features=layers,
            n_out=env.action_dim,
            precision=precision,
            output_bias_init=1.0,
        )
    elif config["use_terminal_value"]:
        neural_net = ActorCritic(
            actions_dim=env.action_dim,
            hidden_dims_actor=layers,
            hidden_dims_critic=config["layers_critic"],
            precision=precision,
        )
    else:
        neural_net = PolicyNet(
            features=layers,
            n_out=env.action_dim,
            precision=precision,
        )
    print("Neural network created successfully.", flush=True)

    def policy_fn(params, obs):
        output = neural_net.apply(params, obs)
        return output[0] if config["use_terminal_value"] else output

    welfare_eval_fn = create_welfare_eval_fn(
        env.econ,
        policy_fn=policy_fn,
        horizon=config.get("welfare_history_horizon", config["welfare_horizon"]),
        n_epis=config["welfare_n_epis"],
        init_range=config["welfare_init_range"],
        init_range_a=config.get("welfare_init_range_a"),
        policy_beta=config.get("policy_beta_end", 500.0) if constraint_mode == "exact_kink" else None,
        hard_floor=True if constraint_mode == "exact_kink" else None,
    )
    if constraint_mode == "learned_multiplier":
        marginal_utility_ss = env.econ.marginal_utility(env.econ.C_ss)

        def combined_policy_fn(params, obs):
            actor_params, multiplier_params = params
            action = neural_net.apply(actor_params, obs)
            multiplier = multiplier_net.apply(multiplier_params, obs)
            K, productivity = env.econ._capital_and_productivity(obs)
            output = env.econ.production(K, productivity)
            _, consumption, _ = env.econ.allocation_from_policy(action, output)
            marginal_utility = env.econ.marginal_utility(consumption)
            deqn_multiplier = multiplier * marginal_utility_ss / marginal_utility
            deqn_multiplier = jnp.maximum(deqn_multiplier, 1e-8)
            multiplier_logit = jnp.where(
                deqn_multiplier > 20,
                deqn_multiplier,
                jnp.log(jnp.expm1(deqn_multiplier)),
            )
            return jnp.concatenate([action, multiplier_logit], axis=-1)

        combined_euler_eval_fn = create_euler_eval_fn(
            env.econ,
            policy_fn=combined_policy_fn,
            periods_per_epis=config["eval_periods_per_epis"],
            n_epis=config["eval_n_epis"],
            mc_draws=DEQN_EVAL_MC_DRAWS,
            init_range=config["init_range"],
            init_range_a=config.get("init_range_a"),
            simul_vol_scale=config["simul_vol_scale"],
        )

        def euler_eval_fn(actor_params, multiplier_params, rng):
            return combined_euler_eval_fn((actor_params, multiplier_params), rng)

    else:
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

    # Create training and evaluation functions
    print("Creating training functions...", flush=True)
    if constraint_mode == "learned_multiplier":
        epoch_train_fn = create_constrained_epoch_train_fn(
            env,
            neural_net.apply,
            multiplier_net.apply,
            config,
        )
        eval_fn = create_constrained_eval_fn(
            env,
            neural_net.apply,
            multiplier_net.apply,
            config,
        )
        multiplier_warmup_fn = create_multiplier_warmup_fn(
            env,
            neural_net.apply,
            multiplier_net.apply,
            config,
        )
    elif constraint_mode == "exact_kink":
        epoch_train_fn = create_exact_kink_epoch_train_fn(env, neural_net.apply, config)
        eval_fn = create_exact_kink_eval_fn(env, neural_net.apply, config)
    else:
        epoch_train_fn = create_epoch_train_fn(env, config)
        eval_fn = create_eval_fn(env, config)

    # Run training
    print("Starting training...", flush=True)
    if constraint_mode == "learned_multiplier":
        result = run_constrained_experiment(
            config=config,
            env=env,
            actor_net=neural_net,
            multiplier_net=multiplier_net,
            epoch_train_fn=epoch_train_fn,
            eval_fn=eval_fn,
            multiplier_warmup_fn=multiplier_warmup_fn,
            welfare_eval_fn=welfare_eval_fn,
            euler_eval_fn=euler_eval_fn,
        )
    else:
        result = run_experiment(
            config=config,
            env=env,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
            eval_fn=eval_fn,
            welfare_eval_fn=welfare_eval_fn,
            euler_eval_fn=euler_eval_fn,
        )

    # Generate plots
    if result and config.get("generate_plots", True) and constraint_mode != "learned_multiplier":
        plots_dir = os.path.join(config["working_dir"], config["run_name"])

        plot_training_metrics(
            training_results=result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100
        )
        plot_learning_rate_schedule(
            training_results=result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100
        )

        if "metrics" in result:
            m = result["metrics"]
            deqn_run = "rbc_irreversible" if config.get("i_min_frac", 0) else "rbc_baseline"
            deqn_results = os.path.join(repo_root, "DEQN", "econ_models", "RBC", "results", deqn_run, "results.json")
            histories = collect_eval_histories("APG", m.get("welfare_history"), "DEQN", deqn_results)
            plot_eval_comparison(
                histories,
                os.path.join(plots_dir, "deqn_vs_apg_eval.png"),
                steps_per_epoch=config["steps_per_epoch"],
            )
            critic_acc = m.get("final_critic_acc")
            acc_text = f"{critic_acc:.2f}%" if critic_acc is not None else "n/a"
            welfare = m.get("welfare")
            welfare_text = ""
            if welfare:
                welfare_text = (
                    f" | CE vs s_ss={100 * welfare['ce_vs_fixed_saving_rate']:+.4f}%" f" | K/Kss={welfare['K_rel']:.4f}"
                )
                if "std_y" in welfare:
                    welfare_text += (
                        f" | std(C)={welfare['std_c']:.4f}"
                        f" std(Y)={welfare['std_y']:.4f}"
                        f" std(I)={welfare['std_i']:.4f}"
                    )
                if "euler_acc" in welfare:
                    welfare_text += f" | Euler={100 * welfare['euler_acc']:.2f}%" f" loss={welfare['euler_loss']:.6f}"
                if "i_bind_frac" in welfare:
                    welfare_text += f" | bind={100 * welfare['i_bind_frac']:.1f}%"
                if "i_violation_frac" in welfare:
                    welfare_text += (
                        f" | violate={100 * welfare['i_violation_frac']:.2f}%"
                        f" mean={welfare['i_violation_mean']:.3e} Iss"
                        f" max={welfare['i_violation_max']:.3e} Iss"
                    )
            print(
                f"Min Loss: {m['min_loss']:.7f} | "
                f"Final Acc: {acc_text} | "
                f"Time: {m['time_fullexp_minutes']:.1f}m"
                f"{welfare_text}"
            )

    return result


if __name__ == "__main__":
    main()
