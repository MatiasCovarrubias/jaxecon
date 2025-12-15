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
import jax.random as random
import optax
import orbax.checkpoint as ocp
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Custom TrainState for compatibility."""

    pass


def run_experiment(config, env, neural_net, epoch_train_fn, eval_fn):
    """
    Run a single APG training experiment with Orbax checkpointing.

    Args:
        config: Configuration dictionary containing all hyperparameters
        env: Environment instance
        neural_net: Neural network instance (Flax module)
        epoch_train_fn: Function that creates the epoch training function
        eval_fn: Function that creates the evaluation function

    Returns:
        dict: Dictionary containing:
            - train_state: Final trained state
            - metrics: Training and evaluation metrics
            - config: Configuration used
    """
    n_cores = len(jax.devices())
    print(f"Running on {n_cores} device(s)")

    # CREATE RNGS
    rng = random.PRNGKey(config["seed"])
    rng, rng_pol, rng_env, rng_epoch, rng_eval = random.split(rng, num=5)

    # CREATE LR SCHEDULE
    if callable(config["learning_rate"]):
        lr_schedule = config["learning_rate"]
    else:
        total_steps = config["n_epochs"] * config["steps_per_epoch"]
        lr_schedule = optax.cosine_decay_schedule(
            init_value=config["learning_rate"],
            decay_steps=total_steps,
            alpha=0.0000001,
        )

    # CREATE OPTIMIZER
    if config.get("max_grad_norm"):
        optim = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(lr_schedule))
    else:
        optim = optax.adam(lr_schedule)

    # INITIALIZE ENV AND TRAIN STATE
    obs, _ = env.reset(rng_env)
    params = neural_net.init(rng_pol, obs)
    train_state_obj = TrainState.create(apply_fn=neural_net.apply, params=params, tx=optim)

    # GET TRAIN AND EVAL FUNCTIONS
    train_epoch_jitted = jax.jit(epoch_train_fn)
    eval_fn_jitted = jax.jit(eval_fn)

    # COMPILE CODE
    print("Compiling functions...")
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    eval_fn_jitted(train_state_obj, rng_eval)
    time_compilation = time() - time_start
    print(f"Compilation completed in {time_compilation:.2f} seconds")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    train_epoch_jitted(train_state_obj, rng_epoch)
    time_epoch = time() - time_start
    print(f"Time per epoch: {time_epoch:.2f} seconds")

    time_start = time()
    eval_fn_jitted(train_state_obj, rng_eval)
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
    learning_rates, checkpointed_steps = [], []

    # CREATE CHECKPOINT DIRECTORY
    checkpoint_dir = Path(config["working_dir"]) / config["run_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))

    # RUN ALL EPOCHS
    print("\nStarting training...")
    time_start = time()
    for i in range(1, config["n_epochs"] + 1):
        train_state_obj, rng_epoch, epoch_metrics = train_epoch_jitted(train_state_obj, rng_epoch)
        eval_metrics = eval_fn_jitted(train_state_obj, rng_eval)

        current_step = i * config["steps_per_epoch"]
        current_lr = lr_schedule(current_step) if callable(lr_schedule) else lr_schedule

        # Extract metrics
        loss_metrics, grad_metrics = epoch_metrics
        mean_loss = float(jnp.mean(loss_metrics[0]))
        mean_actor_loss = float(jnp.mean(loss_metrics[1][0]))
        mean_critic_loss = float(jnp.mean(loss_metrics[1][1]))
        mean_critic_acc = float((1 - jnp.abs(jnp.mean(loss_metrics[1][2]))) * 100)
        mean_grad = float(jnp.mean(grad_metrics[0]))
        max_grad = float(jnp.mean(jnp.max(grad_metrics[1])))

        mean_losses.append(mean_loss)
        mean_actor_losses.append(mean_actor_loss)
        mean_critic_losses.append(mean_critic_loss)
        mean_critic_accs.append(mean_critic_acc)
        mean_grads.append(mean_grad)
        max_grads.append(max_grad)
        learning_rates.append(float(current_lr))
        checkpointed_steps.append(current_step)

        print(
            f"Epoch {i}: Loss={mean_loss:.6f}, Actor={mean_actor_loss:.6f}, "
            f"Critic={mean_critic_loss:.6f}, Acc={mean_critic_acc:.2f}%, LR={current_lr:.6f}"
        )

        print(
            f"  Eval: Loss={eval_metrics[0]:.6f}, Actor={eval_metrics[1]:.6f}, "
            f"Critic={eval_metrics[2]:.6f}, Acc={eval_metrics[3]:.2f}%"
        )

        # Checkpoint
        checkpoint_freq = config.get("checkpoint_every_n_epochs", 10)
        if i >= checkpoint_freq and i % checkpoint_freq == 0:
            checkpoint_manager.save(step=current_step, args=ocp.args.StandardSave(train_state_obj))

    # FINAL SUMMARY
    time_fullexp = (time() - time_start) / 60
    print(f"\nTraining completed in {time_fullexp:.2f} minutes")
    print(f"Minimum loss: {min(mean_losses):.6f}")
    print(f"Final critic accuracy: {mean_critic_accs[-1]:.2f}%")

    # PREPARE RESULTS
    results = {
        "run_name": config["run_name"],
        "min_loss": min(mean_losses),
        "min_actor_loss": min(mean_actor_losses),
        "min_critic_loss": min(mean_critic_losses),
        "final_critic_acc": mean_critic_accs[-1],
        "time_fullexp_minutes": time_fullexp,
        "time_epoch_seconds": time_epoch,
        "time_compilation_seconds": time_compilation,
        "steps_per_second": steps_per_second,
        "n_cores": n_cores,
        "losses": mean_losses,
        "actor_losses": mean_actor_losses,
        "critic_losses": mean_critic_losses,
        "critic_accs": mean_critic_accs,
        "mean_grads": mean_grads,
        "max_grads": max_grads,
        "learning_rates": learning_rates,
        "checkpointed_steps": checkpointed_steps,
    }

    # Save final checkpoint
    checkpoint_manager.save(step=config["n_epochs"] * config["steps_per_epoch"], args=ocp.args.StandardSave(train_state_obj))

    # SAVE RESULTS TO JSON
    with open(checkpoint_dir / "results.json", "w") as f:
        save_data = {"config": {k: v for k, v in config.items() if not callable(v)}, **results}
        json.dump(save_data, f, indent=2)

    return {
        "train_state": train_state_obj,
        "metrics": results,
        "lr_schedule": lr_schedule,
        "config": config,
    }


