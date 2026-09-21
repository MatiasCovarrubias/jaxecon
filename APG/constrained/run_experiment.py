"""Experiment runner for the two-optimizer constrained APG variant."""

import json
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp

from APG.constrained.primal_dual import ConstrainedTrainState
from DEQN.econ_models.RBC.euler_eval import euler_metrics_to_dict, print_euler_metrics
from DEQN.econ_models.RBC.train_shared import (
    create_optimizer,
    init_policy_params,
    save_policy_params,
)
from DEQN.econ_models.RBC.welfare_eval import (
    print_welfare_metrics,
    welfare_metrics_to_dict,
)


def _objective_to_dict(metrics):
    return {name: float(value) for name, value in zip(metrics._fields, metrics)}


def run_constrained_experiment(
    config,
    env,
    actor_net,
    multiplier_net,
    epoch_train_fn,
    eval_fn,
    multiplier_warmup_fn,
    welfare_eval_fn,
    euler_eval_fn,
):
    """Train and serialize independent actor and multiplier networks."""
    rng = jax.random.PRNGKey(config["seed"])
    (
        rng,
        actor_rng,
        multiplier_rng,
        epoch_rng,
        warmup_rng,
        eval_rng,
        welfare_rng,
        euler_rng,
    ) = jax.random.split(rng, 8)

    actor_steps_per_update = int(
        config.get("actor_steps_per_multiplier_update", 1)
    )
    actor_optimizer_config = {
        **config,
        "steps_per_epoch": (
            int(config["steps_per_epoch"]) * actor_steps_per_update
        ),
    }
    actor_tx, actor_lr_schedule = create_optimizer(actor_optimizer_config)
    multiplier_config = {
        **config,
        "learning_rate": config["multiplier_learning_rate"],
        "max_grad_norm": config.get(
            "multiplier_max_grad_norm",
            config.get("max_grad_norm"),
        ),
    }
    if config.get("multiplier_update_mode") == "phr_direct":
        multiplier_tx = optax.set_to_zero()
        multiplier_lr_schedule = lambda _: jnp.asarray(
            0.0,
            dtype=env.econ.precision,
        )
    else:
        multiplier_tx, multiplier_lr_schedule = create_optimizer(
            multiplier_config
        )
    actor_params = init_policy_params(actor_net, actor_rng, env.obs_ss)
    multiplier_params = init_policy_params(multiplier_net, multiplier_rng, env.obs_ss)
    state = ConstrainedTrainState(
        actor=train_state.TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_params,
            tx=actor_tx,
        ),
        multiplier=train_state.TrainState.create(
            apply_fn=multiplier_net.apply,
            params=multiplier_params,
            tx=multiplier_tx,
        ),
    )

    train_epoch_jitted = jax.jit(epoch_train_fn)
    eval_fn_jitted = jax.jit(eval_fn)
    warmup_jitted = jax.jit(multiplier_warmup_fn)
    print("Compiling constrained functions...")
    start = time()
    train_epoch_jitted(state, epoch_rng)
    if config.get("multiplier_warmup_steps", 0):
        state, warmup_rng, warmup_metrics = warmup_jitted(state, warmup_rng)
        print(
            "Multiplier warmup completed: "
            f"steps={config['multiplier_warmup_steps']} "
            f"lambda={float(warmup_metrics.multiplier_mean[-1]):.4g}"
        )
    eval_fn_jitted(state, eval_rng)
    welfare_eval_fn(state.actor.params, welfare_rng)
    euler_eval_fn(state.actor.params, state.multiplier.params, euler_rng)
    compilation_seconds = time() - start
    print(f"Compilation completed in {compilation_seconds:.2f} seconds")

    checkpoint_dir = Path(config["working_dir"]) / config["run_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = None
    if config.get("save_orbax_checkpoint", True):
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

    epochs = []
    actor_learning_rates = []
    multiplier_learning_rates = []
    train_history = []
    eval_history = []
    welfare_history = []
    final_welfare = None
    welfare_every = int(config.get("welfare_every_n_epochs", 5))
    checkpoint_every = int(config.get("checkpoint_every_n_epochs", 10))
    last_saved_step = None
    select_best = bool(config.get("select_best_checkpoint", False))
    selection_penalty = float(config.get("selection_violation_penalty", 10.0))
    selection_euler_weight = float(config.get("selection_euler_weight", 0.25))
    max_occupancy_violation = float(
        config.get("selection_max_occupancy_violation_frac", 0.02)
    )
    max_grid_violation = float(
        config.get("selection_max_grid_violation_frac", 0.01)
    )
    best_state = None
    best_checkpoint = None
    best_model_metrics = None

    def record_eval(epoch):
        metrics = eval_fn_jitted(state, eval_rng)
        row = {
            "epoch": epoch,
            **_objective_to_dict(metrics.objective),
            "actor_grad_norm": float(metrics.actor_grad_norm),
            "multiplier_grad_norm": float(metrics.multiplier_grad_norm),
        }
        eval_history.append(row)
        return row

    def record_model_diagnostics(epoch, eval_row):
        nonlocal final_welfare, best_state, best_checkpoint, best_model_metrics
        row = {"epoch": epoch}
        euler_metrics = euler_eval_fn(
            state.actor.params,
            state.multiplier.params,
            euler_rng,
        )
        print_euler_metrics(euler_metrics)
        row.update(euler_metrics_to_dict(euler_metrics))
        welfare_metrics = welfare_eval_fn(state.actor.params, welfare_rng)
        print_welfare_metrics(welfare_metrics)
        row.update(welfare_metrics_to_dict(welfare_metrics))
        welfare_history.append(row)
        final_welfare = {key: value for key, value in row.items() if key != "epoch"}
        occupancy_excess = max(
            eval_row["occupancy_violation_frac"] - max_occupancy_violation,
            0.0,
        )
        grid_excess = max(
            eval_row["grid_violation_frac"] - max_grid_violation,
            0.0,
        )
        selection_score = (
            selection_euler_weight * row["euler_loss"]
            - row["ce_vs_fixed_saving_rate"]
            + selection_penalty * (occupancy_excess + grid_excess)
        )
        if (
            select_best
            and (
                best_checkpoint is None
                or selection_score < best_checkpoint["score"]
            )
        ):
            best_state = state
            best_model_metrics = dict(final_welfare)
            best_checkpoint = {
                "epoch": epoch,
                "score": selection_score,
                "euler_loss": row["euler_loss"],
                "euler_acc": row["euler_acc"],
                "euler_min_acc": row["euler_min_acc"],
                "ce_vs_fixed_saving_rate": row[
                    "ce_vs_fixed_saving_rate"
                ],
                "occupancy_violation_frac": eval_row[
                    "occupancy_violation_frac"
                ],
                "grid_violation_frac": eval_row["grid_violation_frac"],
                "occupancy_shortfall_mean": eval_row[
                    "occupancy_shortfall_mean"
                ],
                "grid_shortfall_mean": eval_row["grid_shortfall_mean"],
            }

    print("\nStarting constrained training...")
    training_start = time()
    initial_eval = record_eval(0)
    epochs.append(0)
    actor_learning_rates.append(float(actor_lr_schedule(0)))
    multiplier_learning_rates.append(float(multiplier_lr_schedule(0)))
    print(
        "Epoch 0 (init): "
        f"Actor={initial_eval['actor_loss']:.6f} "
        f"Dual={initial_eval['dual_loss']:.6f} "
        f"Grid violate={100 * initial_eval['grid_violation_frac']:.2f}%"
    )
    record_model_diagnostics(0, initial_eval)

    for epoch in range(1, int(config["n_epochs"]) + 1):
        state, epoch_rng, step_metrics = train_epoch_jitted(state, epoch_rng)
        objective = jax.tree_util.tree_map(jnp.mean, step_metrics.objective)
        train_row = {
            "epoch": epoch,
            **_objective_to_dict(objective),
            "actor_grad_norm": float(jnp.mean(step_metrics.actor_grad_norm)),
            "multiplier_grad_norm": float(
                jnp.mean(step_metrics.multiplier_grad_norm)
            ),
        }
        train_history.append(train_row)
        eval_row = record_eval(epoch)
        current_step = epoch * int(config["steps_per_epoch"])
        current_actor_step = current_step * actor_steps_per_update
        actor_lr = float(actor_lr_schedule(current_actor_step))
        multiplier_lr = float(multiplier_lr_schedule(current_step))
        epochs.append(epoch)
        actor_learning_rates.append(actor_lr)
        multiplier_learning_rates.append(multiplier_lr)
        print(
            f"Epoch {epoch}: Actor={train_row['actor_loss']:.6f} "
            f"Dual={train_row['dual_loss']:.6f} "
            f"LR=({actor_lr:.6g}, {multiplier_lr:.6g})"
        )
        print(
            f"  Eval: true={eval_row['true_return']:.6f} "
            f"aug={eval_row['augmented_return']:.6f} "
            f"violate=({100 * eval_row['occupancy_violation_frac']:.2f}%, "
            f"{100 * eval_row['grid_violation_frac']:.2f}%) "
            f"lambda={eval_row['multiplier_mean']:.4g}"
        )

        if epoch == config["n_epochs"] or epoch % welfare_every == 0:
            record_model_diagnostics(epoch, eval_row)
        if (
            checkpoint_manager is not None
            and epoch >= checkpoint_every
            and epoch % checkpoint_every == 0
            and (not select_best or epoch < int(config["n_epochs"]))
        ):
            checkpoint_manager.save(
                step=current_step,
                args=ocp.args.StandardSave(state),
            )
            last_saved_step = current_step

    training_minutes = (time() - training_start) / 60
    if select_best and best_state is not None:
        state = best_state
        final_welfare = best_model_metrics
        print(
            "Selected checkpoint: "
            f"epoch={best_checkpoint['epoch']} "
            f"score={best_checkpoint['score']:.6g} "
            f"Euler={best_checkpoint['euler_loss']:.6g} "
            f"CE={100 * best_checkpoint['ce_vs_fixed_saving_rate']:+.4f}% "
            f"violate=({100 * best_checkpoint['occupancy_violation_frac']:.2f}%, "
            f"{100 * best_checkpoint['grid_violation_frac']:.2f}%)"
        )
    final_step = int(config["n_epochs"]) * int(config["steps_per_epoch"])
    if checkpoint_manager is not None:
        if last_saved_step != final_step:
            checkpoint_manager.save(
                step=final_step,
                args=ocp.args.StandardSave(state),
            )
        checkpoint_manager.wait_until_finished()

    results = {
        "run_name": config["run_name"],
        "algorithm_variant": config.get(
            "algorithm_schema",
            "constrained_apg_learned_multiplier_v1",
        ),
        "time_fullexp_minutes": training_minutes,
        "time_compilation_seconds": compilation_seconds,
        "epochs": epochs,
        "learning_rates": actor_learning_rates,
        "multiplier_learning_rates": multiplier_learning_rates,
        "train_history": train_history,
        "eval_history": eval_history,
        "welfare": final_welfare,
        "welfare_history": welfare_history,
        "selected_checkpoint": best_checkpoint,
    }
    save_data = {
        "config": {key: value for key, value in config.items() if not callable(value)},
        **results,
    }
    with open(checkpoint_dir / "results.json", "w") as handle:
        json.dump(save_data, handle, indent=2)
    save_policy_params(state.actor.params, checkpoint_dir / "params.msgpack")
    save_policy_params(
        state.multiplier.params,
        checkpoint_dir / "multiplier_params.msgpack",
    )
    print(f"\nConstrained training completed in {training_minutes:.2f} minutes")
    return {
        "train_state": state,
        "metrics": results,
        "lr_schedule": actor_lr_schedule,
        "multiplier_lr_schedule": multiplier_lr_schedule,
        "config": config,
    }
