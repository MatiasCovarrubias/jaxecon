#!/usr/bin/env python3
"""
APG Training script.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m APG.train

        # Method 2: Run directly as script (from repository root):
        python APG/train.py

        Both methods require you to be in the repository root directory.

    COLAB:
        Simply run all cells in order. The script will automatically detect the Colab
        environment, install dependencies, clone the repository, and mount Google Drive.
"""

import os
import sys

# ============================================================================
# ENVIRONMENT DETECTION AND SETUP
# ============================================================================

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    print("Installing JAX with CUDA support...")
    import subprocess

    subprocess.run(["pip", "install", "--upgrade", "jax[cuda12]"], check=True)
    subprocess.run(["pip", "install", "orbax-checkpoint"], check=True)

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    sys.path.insert(0, "/content/jaxecon")

    print("Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    base_dir = "/content/drive/MyDrive/Jaxecon/APG"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "APG")

# ============================================================================
# IMPORTS
# ============================================================================

import optax  # noqa: E402
from flax import linen as nn  # noqa: E402
from jax import config as jax_config  # noqa: E402

from APG.algorithm import create_epoch_train_fn, create_eval_fn  # noqa: E402
from APG.environments import RbcMultiSector  # noqa: E402
from APG.neural_nets import ActorCritic  # noqa: E402
from APG.training import run_experiment  # noqa: E402
from APG.training.plots import plot_training_metrics, plot_learning_rate_schedule  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_lr_schedule():
    """Create learning rate schedule."""
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(0, 0.01, 100),
            optax.constant_schedule(0.01),
            optax.constant_schedule(0.001),
            optax.constant_schedule(0.0001),
            optax.cosine_decay_schedule(0.0001, 1000),
        ],
        boundaries=[200, 400, 600, 800],
    )


config = {
    # Key configuration - Edit these first
    "run_name": "rbc_ms_baseline",
    "date": "Dec2025",
    "seed": 42,
    # Environment parameters
    "n_sectors": 8,
    # Training parameters
    "fp64_precision": False,
    "learning_rate": get_lr_schedule(),
    "n_epochs": 100,
    "steps_per_epoch": 100,
    "epis_per_step": 1024 * 8,
    "periods_per_epis": 32,
    "checkpoint_every_n_epochs": 10,
    # Evaluation parameters
    "eval_n_epis": 1024 * 32,
    "eval_periods_per_epis": 32,
    # Algorithm parameters
    "gae_lambda": 0.95,
    "max_grad_norm": None,
    # Neural network architecture
    "layers_actor": [16, 8],
    "layers_critic": [8, 4],
    # Directories
    "working_dir": os.path.join(base_dir, "results/"),
}


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    print(f"Training: {config['run_name']}", flush=True)

    # Precision setup
    if config["fp64_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Create environment
    print("Creating environment...", flush=True)
    env = RbcMultiSector(N=config["n_sectors"])
    print(f"  n_sectors: {config['n_sectors']}", flush=True)
    print(f"  obs_dim: {env.obs_dim}", flush=True)
    print(f"  action_dim: {env.action_dim}", flush=True)

    # Create neural network
    print("Creating neural network...", flush=True)
    neural_net = ActorCritic(
        actions_dim=env.action_dim,
        hidden_dims_actor=config["layers_actor"],
        hidden_dims_critic=config["layers_critic"],
        activation_final_actor=nn.softmax,
    )
    print("Neural network created successfully.", flush=True)

    # Create training and evaluation functions
    print("Creating training functions...", flush=True)
    epoch_train_fn = create_epoch_train_fn(env, config)
    eval_fn = create_eval_fn(env, config)

    # Run training
    print("Starting training...", flush=True)
    try:
        result = run_experiment(
            config=config,
            env=env,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
            eval_fn=eval_fn,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Generate plots
    if result:
        plots_dir = os.path.join(config["working_dir"], config["run_name"])

        plot_training_metrics(
            training_results=result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100
        )
        plot_learning_rate_schedule(
            training_results=result, save_dir=plots_dir, experiment_name=config["run_name"], display_dpi=100
        )

        if "metrics" in result:
            m = result["metrics"]
            print(
                f"Min Loss: {m['min_loss']:.7f} | "
                f"Final Acc: {m['final_critic_acc']:.2f}% | "
                f"Time: {m['time_fullexp_minutes']:.1f}m"
            )

    return result


if __name__ == "__main__":
    main()


