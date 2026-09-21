"""
Experiment runner for APG training.

Provides high-level orchestration of training experiments including:
- Neural network initialization
- Learning rate scheduling
- Training loop execution
- Checkpointing with Orbax
- Metrics collection
"""

import json
from pathlib import Path
from time import time

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state

from DEQN.econ_models.RBC.train_shared import (
    create_optimizer,
    init_policy_params,
    maybe_save_policy_snapshot,
    resolve_policy_snapshot_epochs,
    save_policy_params,
    split_shared_rbc_rngs,
)
from DEQN.econ_models.RBC.welfare_eval import print_welfare_metrics, welfare_metrics_to_dict
from DEQN.econ_models.RBC.euler_eval import euler_metrics_to_dict, print_euler_metrics


class TrainState(train_state.TrainState):
    """Custom TrainState for compatibility."""

    pass


def run_experiment(config, env, neural_net, epoch_train_fn, eval_fn, welfare_eval_fn=None, euler_eval_fn=None):
    """
    Run a single APG training experiment with Orbax checkpointing.

    Args:
        config: Configuration dictionary containing all hyperparameters.
            Optional ``policy_snapshot_every_n_epochs`` / ``policy_snapshot_epochs``
            write portable ``params_epoch_*.msgpack`` files (epoch 0 and final
            included when enabled). Off by default.
        env: Environment instance
        neural_net: Neural network instance (Flax module)
        epoch_train_fn: Function that creates the epoch training function
        eval_fn: Evaluation function
        welfare_eval_fn: Optional (params, rng) -> WelfareMetrics from the shared RBC helper
        euler_eval_fn: Optional (params, rng) -> (mse, acc, min_acc) Euler diagnostic

    Returns:
        dict: Dictionary containing:
            - train_state: Final trained state
            - metrics: Training and evaluation metrics
            - config: Configuration used
    """
    use_terminal_value = config.get("use_terminal_value", False)
    n_cores = len(jax.devices())
    print(f"Running on {n_cores} device(s)")

    # CREATE RNGS. Shared 5-way split with DEQN; Euler eval is APG-only.
    streams = split_shared_rbc_rngs(config["seed"], extra_names=("euler",))
    rng_pol = streams["init"]
    rng_epoch = streams["epoch"]
    rng_eval = streams["eval"]
    rng_welfare = streams["welfare"]
    rng_euler = streams["euler"]

    tx, lr_schedule = create_optimizer(config)
    params = init_policy_params(neural_net, rng_pol, env.obs_ss)
    train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=tx)

    # GET TRAIN AND EVAL FUNCTIONS
    train_epoch_jitted = jax.jit(epoch_train_fn)
    eval_fn_jitted = jax.jit(eval_fn)
    welfare_every = config.get("welfare_every_n_epochs", 5)

    # COMPILE CODE
    print("Compiling functions...")
    time_start = time()
    jax.block_until_ready(train_epoch_jitted(train_state_obj, rng_epoch))
    jax.block_until_ready(eval_fn_jitted(train_state_obj, rng_eval))
    if welfare_eval_fn is not None:
        jax.block_until_ready(welfare_eval_fn(train_state_obj.params, rng_welfare))
    if euler_eval_fn is not None:
        jax.block_until_ready(euler_eval_fn(train_state_obj.params, rng_euler))
    time_compilation = time() - time_start
    print(f"Compilation completed in {time_compilation:.2f} seconds")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    jax.block_until_ready(train_epoch_jitted(train_state_obj, rng_epoch))
    time_epoch = time() - time_start
    print(f"Time per epoch: {time_epoch:.2f} seconds")

    time_start = time()
    jax.block_until_ready(eval_fn_jitted(train_state_obj, rng_eval))
    time_eval = time() - time_start
    print(f"Time per evaluation: {time_eval:.2f} seconds")

    time_experiment = (time_epoch + time_eval) * config["n_epochs"] / 60
    print(f"Estimated experiment time: {time_experiment:.2f} minutes")

    steps_per_second = (
        n_cores * config["steps_per_epoch"] * config["epis_per_step"] * config["periods_per_epis"] / time_epoch
    )
    print(f"Steps per second: {steps_per_second:.2f}")

    # CREATE LISTS TO STORE METRICS
    mean_losses, mean_actor_losses, mean_critic_losses = [], [], []
    mean_critic_accs, mean_grads, max_grads = [], [], []
    eval_losses, eval_actor_losses = [], []
    learning_rates, checkpointed_steps, epochs = [], [], []
    welfare_history = []
    final_welfare = None

    def record_model_diagnostics(epoch):
        nonlocal final_welfare
        row = {"epoch": epoch}
        if euler_eval_fn is not None:
            euler_metrics = euler_eval_fn(train_state_obj.params, rng_euler)
            print_euler_metrics(euler_metrics)
            row.update(euler_metrics_to_dict(euler_metrics))
        if welfare_eval_fn is not None:
            welfare_metrics = welfare_eval_fn(train_state_obj.params, rng_welfare)
            print_welfare_metrics(welfare_metrics)
            row.update(welfare_metrics_to_dict(welfare_metrics))
        if len(row) > 1:
            welfare_history.append(row)
            final_welfare = {key: value for key, value in row.items() if key != "epoch"}
        return row

    def record_eval_and_schedule(epoch, eval_metrics):
        current_step = epoch * config["steps_per_epoch"]
        current_lr = lr_schedule(current_step) if callable(lr_schedule) else lr_schedule
        eval_losses.append(float(eval_metrics[0]))
        eval_actor_losses.append(float(eval_metrics[1]))
        learning_rates.append(float(current_lr))
        checkpointed_steps.append(current_step)
        epochs.append(epoch)
        return current_step, current_lr

    # CREATE CHECKPOINT DIRECTORY
    checkpoint_dir = Path(config["working_dir"]) / config["run_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = None
    if config.get("save_orbax_checkpoint", True):
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )
    last_saved_step = None
    snapshot_epochs = resolve_policy_snapshot_epochs(config)

    # RUN ALL EPOCHS
    print("\nStarting training...")
    time_start = time()

    eval_metrics = eval_fn_jitted(train_state_obj, rng_eval)
    current_step, current_lr = record_eval_and_schedule(0, eval_metrics)
    print(f"Epoch 0 (init): Eval Loss={eval_metrics[0]:.6f}, Actor={eval_metrics[1]:.6f}, LR={current_lr:.6f}")
    record_model_diagnostics(0)
    maybe_save_policy_snapshot(train_state_obj.params, checkpoint_dir, 0, snapshot_epochs)

    for i in range(1, config["n_epochs"] + 1):
        train_state_obj, rng_epoch, epoch_metrics = train_epoch_jitted(train_state_obj, rng_epoch)
        eval_metrics = eval_fn_jitted(train_state_obj, rng_eval)
        current_step, current_lr = record_eval_and_schedule(i, eval_metrics)

        loss_metrics, grad_metrics = epoch_metrics
        mean_loss = float(jnp.mean(loss_metrics[0]))
        mean_actor_loss = float(jnp.mean(loss_metrics[1][0]))
        mean_grad = float(jnp.mean(grad_metrics[0]))
        max_grad = float(jnp.mean(jnp.max(grad_metrics[1])))

        mean_losses.append(mean_loss)
        mean_actor_losses.append(mean_actor_loss)
        mean_grads.append(mean_grad)
        max_grads.append(max_grad)

        if use_terminal_value:
            mean_critic_loss = float(jnp.mean(loss_metrics[1][1]))
            mean_critic_acc = float((1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100)
            mean_critic_losses.append(mean_critic_loss)
            mean_critic_accs.append(mean_critic_acc)
            print(
                f"Epoch {i}: Loss={mean_loss:.6f}, Actor={mean_actor_loss:.6f}, "
                f"Critic={mean_critic_loss:.6f}, Acc={mean_critic_acc:.2f}%, LR={current_lr:.6f}"
            )
            print(
                f"  Eval: Loss={eval_metrics[0]:.6f}, Actor={eval_metrics[1]:.6f}, "
                f"Critic={eval_metrics[2]:.6f}, Acc={eval_metrics[3]:.2f}%"
            )
        else:
            print(f"Epoch {i}: Loss={mean_loss:.6f}, Actor={mean_actor_loss:.6f}, LR={current_lr:.6f}")
            print(f"  Eval: Loss={eval_metrics[0]:.6f}, Actor={eval_metrics[1]:.6f}")

        if welfare_eval_fn is not None or euler_eval_fn is not None:
            if i == config["n_epochs"] or i % welfare_every == 0:
                record_model_diagnostics(i)

        maybe_save_policy_snapshot(train_state_obj.params, checkpoint_dir, i, snapshot_epochs)

        # Checkpoint
        checkpoint_freq = config.get("checkpoint_every_n_epochs", 10)
        if checkpoint_manager is not None and i >= checkpoint_freq and i % checkpoint_freq == 0:
            checkpoint_manager.save(step=current_step, args=ocp.args.StandardSave(train_state_obj))
            last_saved_step = current_step

    # FINAL SUMMARY
    time_fullexp = (time() - time_start) / 60
    print(f"\nTraining completed in {time_fullexp:.2f} minutes")
    print(f"Minimum loss: {min(mean_losses):.6f}")
    if use_terminal_value:
        print(f"Final critic accuracy: {mean_critic_accs[-1]:.2f}%")

    # PREPARE RESULTS
    results = {
        "run_name": config["run_name"],
        "min_loss": min(mean_losses),
        "min_actor_loss": min(mean_actor_losses),
        "min_critic_loss": min(mean_critic_losses) if mean_critic_losses else None,
        "final_critic_acc": mean_critic_accs[-1] if mean_critic_accs else None,
        "time_fullexp_minutes": time_fullexp,
        "time_epoch_seconds": time_epoch,
        "time_compilation_seconds": time_compilation,
        "steps_per_second": steps_per_second,
        "n_cores": n_cores,
        "losses": mean_losses,
        "actor_losses": mean_actor_losses,
        "eval_losses": eval_losses,
        "eval_actor_losses": eval_actor_losses,
        "critic_losses": mean_critic_losses,
        "critic_accs": mean_critic_accs,
        "mean_grads": mean_grads,
        "max_grads": max_grads,
        "learning_rates": learning_rates,
        "checkpointed_steps": checkpointed_steps,
        "epochs": epochs,
        "welfare": final_welfare,
        "welfare_history": welfare_history,
    }

    # Save final checkpoint
    final_step = config["n_epochs"] * config["steps_per_epoch"]
    if checkpoint_manager is not None:
        if last_saved_step != final_step:
            checkpoint_manager.save(step=final_step, args=ocp.args.StandardSave(train_state_obj))
        checkpoint_manager.wait_until_finished()

    # SAVE RESULTS TO JSON
    with open(checkpoint_dir / "results.json", "w") as f:
        save_data = {"config": {k: v for k, v in config.items() if not callable(v)}, **results}
        json.dump(save_data, f, indent=2)
    save_policy_params(train_state_obj.params, checkpoint_dir / "params.msgpack")

    return {
        "train_state": train_state_obj,
        "metrics": results,
        "lr_schedule": lr_schedule,
        "config": config,
    }


