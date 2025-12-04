# APG Algorithm with Modular Components
# This script trains a neural net to output the optimal policy of a nonlinear RBC model using the Analytical Policy Gradient (APG) algorithm. The code has been organized into modular components.

# Standard imports
import json
import os
from time import time

import flax
import jax
import optax
from algorithm.epoch_train import get_apg_train_fn
from algorithm.eval import get_eval_fn
from environments.RbcMultiSector import RbcMultiSector
from flax import linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random

# Import our modular components
from neural_nets.neural_nets import ActorCritic
from utilities.plot_results import plot_results

print("JAX devices:", jax.devices())
print("JAX version:", jax.__version__)
print("Flax version:", flax.__version__)
print("Optax version:", optax.__version__)


# Configuration
def get_config():
    """Get the experiment configuration."""
    # CREATE LEARNING RATE SCHEDULE
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0, 0.01, 100),
            optax.constant_schedule(0.01),
            optax.constant_schedule(0.001),
            optax.constant_schedule(0.0001),
            optax.cosine_decay_schedule(0.0001, 1000),
        ],
        boundaries=[200, 400, 600, 800],
    )

    config_apg = {
        "learning_rate": lr_schedule,
        "n_epochs": 100,
        "steps_per_epoch": 100,
        "epis_per_step": 1024 * 8,
        "periods_per_epis": 32,
        "eval_n_epis": 1024 * 32,
        "eval_periods_per_epis": 32,
        "gae_lambda": 0.95,
        "max_grad_norm": None,
        "layers_actor": [16, 8],
        "layers_critic": [8, 4],
        "seed": 42,
        "fp64_precision": False,
        "run_name": "apg_RbcMS_modular",
        "date": "modular_implementation",
        "working_dir": "./results/",
    }

    print("Configuration loaded successfully")
    return config_apg


# Main Experiment Function
def run_experiment(env, config):
    """Runs experiment."""

    print("Starting experiment...\n")
    if config["fp64_precision"]:
        from jax.config import config as config_jax

        config_jax.update("jax_enable_x64", True)

    n_cores = len(jax.devices())

    # CREATE NN, RNGS, TRAIN_STATE AND EPOCH UPDATE
    nn_policy = ActorCritic(
        actions_dim=env.action_dim,
        hidden_dims_actor=config["layers_actor"],
        hidden_dims_critic=config["layers_critic"],
        activation_final_actor=nn.softmax,
    )

    if config["max_grad_norm"]:
        optim = optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]), optax.adam(config["learning_rate"]))
    else:
        optim = optax.chain(optax.adam(config["learning_rate"]))

    print("Neural Net and Optimizer Created...\n")

    # INITIALIZE ENV AND ALGO STATES
    rng, rng_pol, rng_env, rng_epoch, rng_eval = random.split(random.PRNGKey(config["seed"]), num=5)

    obs, env_state = env.reset(rng_env)
    train_state = TrainState.create(apply_fn=nn_policy.apply, params=nn_policy.init(rng_pol, obs), tx=optim)

    # GET EPOCH TRAIN AND EVAL FUNCTIONS
    epoch_update = jax.jit(get_apg_train_fn(env, config))
    eval_fn = jax.jit(get_eval_fn(env, config))

    # COMPILE CODE
    print("Starting compilation...\n")
    time_start = time()
    epoch_update(train_state, rng_epoch)  # compiles
    eval_fn(train_state, rng_eval)
    time_compilation = time() - time_start
    print("Time Elapsed for Compilation:", time_compilation, "seconds")

    print("Compilation completed. Proceeding to run an epoch and calculate performance statistics...\n")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    epoch_update(train_state, rng_epoch)
    time_epoch = time() - time_start
    print("Time Elapsed for Epoch:", time_epoch, "seconds")
    print(
        "Steps per second:",
        n_cores * config["steps_per_epoch"] * config["epis_per_step"] * config["periods_per_epis"] / time_epoch,
        "st/s",
    )

    # RUN AN EVAL TO GET TIME STATS
    time_start = time()
    eval_fn(train_state, rng_eval)
    time_eval = time() - time_start
    print("Time Elapsed for Eval:", time_eval, "seconds")

    print("Estimated time for full experiment", (time_epoch + time_eval) * config["n_epochs"] / 60, "minutes\n")

    print("Proceeding to run all epochs...\n")

    # CREATE LISTS TO STORE METRICS
    mean_losses, mean_actor_losses, mean_critic_losses, mean_critic_accs, mean_grads, max_grads = [], [], [], [], [], []

    # RUN ALL THE EPOCHS
    time_start = time()
    for i in range(1, config["n_epochs"] + 1):
        train_state, rng_epoch, epoch_metrics = epoch_update(train_state, rng_epoch)
        eval_metrics = eval_fn(train_state, rng_eval)

        mean_losses.append(float(jnp.mean(epoch_metrics[0][0])))
        mean_actor_losses.append(float(jnp.mean(epoch_metrics[0][1][0])))
        mean_critic_losses.append(float(jnp.mean(epoch_metrics[0][1][1])))
        mean_critic_accs.append(float((1 - jnp.abs(jnp.mean(epoch_metrics[0][1][2]))) * 100))
        mean_grads.append(float(jnp.mean(epoch_metrics[1][0])))
        max_grads.append(float(jnp.mean(jnp.max(epoch_metrics[1][1]))))

        print(
            "Iteration:",
            i * config["steps_per_epoch"],
            ", Mean_loss:",
            jnp.mean(epoch_metrics[0][0]),
            ", Mean_actor_loss:",
            jnp.mean(epoch_metrics[0][1][0]),
            ", Mean_critic_loss:",
            jnp.mean(epoch_metrics[0][1][1]),
            ", Mean_critic_acc:",
            (1 - jnp.abs(jnp.mean(epoch_metrics[0][1][2]))) * 100,
            ", Mean_grads:",
            jnp.mean(epoch_metrics[1][0]),
            ", Max_grads:",
            jnp.max(epoch_metrics[1][1]),
            ", Learning rate:",
            config["learning_rate"](i * config["steps_per_epoch"]),
            "\n",
        )

        print(
            "Evaluation:     ",
            ", Mean_loss:",
            eval_metrics[0],
            ", Mean_actor_loss:",
            eval_metrics[1],
            ", Mean_critic_loss:",
            eval_metrics[2],
            ", Mean_critic_acc:",
            eval_metrics[3],
            ", Mean_grads:",
            eval_metrics[4],
            ", Max_grads:",
            eval_metrics[5],
            "\n",
        )

    # STORE RESULTS
    print("Minimum loss attained in training:", min(mean_losses))

    time_fullexp = (time() - time_start) / 60
    print("Time Elapsed for Full Experiment:", time_fullexp, "minutes")

    results = {
        "min_loss": min(mean_losses),
        "min_actor_loss": min(mean_actor_losses),
        "min_critic_loss": min(mean_critic_losses),
        "last_critic_accs": mean_critic_accs[-1],
        "Time for Full Experiment (m)": time_fullexp,
        "Time for epoch (s)": time_epoch,
        "Time for Compilation (s)": time_compilation,
        "Steps per second": n_cores * config["steps_per_epoch"] * config["periods_per_epis"] / time_epoch,
        "n_cores": n_cores,
        "periods_per_epis": config["periods_per_epis"],
        "epis_per_step": config["epis_per_step"],
        "steps_per_epoch": config["steps_per_epoch"],
        "n_epochs": config["n_epochs"],
        "layers_actor": config["layers_actor"],
        "layers_critic": config["layers_critic"],
        "date": config["date"],
        "seed": config["seed"],
        "Losses_list": mean_losses,
        "Actor_losses_list": mean_actor_losses,
        "Critic_losses_list": mean_critic_losses,
        "Critic_accs_list": mean_critic_accs,
        "Mean_grads_list": mean_grads,
        "Max_grads_list": max_grads,
    }

    # Create results directory if it doesn't exist
    if not os.path.exists(config["working_dir"]):
        os.makedirs(config["working_dir"])
    if not os.path.exists(config["working_dir"] + config["run_name"]):
        os.mkdir(config["working_dir"] + config["run_name"])

    # Save results
    with open(config["working_dir"] + config["run_name"] + "/results.json", "w") as write_file:
        json.dump(results, write_file)

    # Store checkpoint
    checkpoints.save_checkpoint(
        ckpt_dir=config["working_dir"] + config["run_name"],
        target=train_state,
        step=config["n_epochs"] * config["steps_per_epoch"],
    )

    return train_state, results


def print_summary(results, config_apg):
    """Print experiment summary."""
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Minimum loss achieved: {results['min_loss']:.6f}")
    print(f"Final critic accuracy: {results['last_critic_accs']:.2f}%")
    print(f"Total experiment time: {results['Time for Full Experiment (m)']:.2f} minutes")
    print(f"Steps per second: {results['Steps per second']:.0f}")
    print(f"Results saved to: {config_apg['working_dir']}{config_apg['run_name']}")


def main():
    """Main function to run the complete experiment."""

    # Get configuration
    config_apg = get_config()

    # Run the main experiment
    final_train_state, results = run_experiment(RbcMultiSector(N=8), config_apg)

    # Plot the training results
    plot_results(config_apg, results)

    # Print summary
    print_summary(results, config_apg)


if __name__ == "__main__":
    main()
