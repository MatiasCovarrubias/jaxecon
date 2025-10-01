#!/usr/bin/env python3
"""
RBC ProdNet Model Training Script - Single Experiment

This script provides a simplified training framework for neural network policies
for the RBC Production Network model using the DEQN solver. Unlike the multi-experiment
version, this script is designed to run a single experiment with a specific configuration,
making it easier to understand and a better entry point for new users.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.econ_models.RbcProdNetv2.train_single_sep23_2025

        # Method 2: Run directly as script (from repository root):
        python DEQN/econ_models/RbcProdNetv2/train_single_sep23_2025.py

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

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    sys.path.insert(0, "/content/jaxecon")

    print("Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    model_dir = "/content/drive/MyDrive/Jaxecon/RbcProdNet/"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    model_dir = os.path.join(repo_root, "DEQN", "econ_models", "RbcProdNetv2")

# ============================================================================
# IMPORTS
# ============================================================================

import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.algorithm import create_fast_epoch_train_fn  # noqa: E402
from DEQN.econ_models.RbcProdNetv2.plots import (  # noqa: E402
    plot_learning_rate_schedule,
    plot_training_metrics,
    plot_training_summary,
)
from DEQN.econ_models.RbcProdNetv2.RbcProdNet_Sept23_2025 import Model  # noqa: E402
from DEQN.neural_nets.neural_nets import create_neural_net_builder  # noqa: E402
from DEQN.training.runner import run_experiment  # noqa: E402

jax_config.update("jax_debug_nans", True)


def create_config(model_name, save_dir, exper_name="baseline_single", seed=7):
    """Create configuration for the training experiment."""
    config = {
        "model_name": model_name,
        "exper_name": exper_name,
        "date": "Sep25_2025",
        "save_dir": save_dir,
        "restore": False,
        "restore_exper_name": None,
        "seed": seed,
        "comment": "Single experiment baseline configuration.",
        "comment_at_end": False,
        "layers": [512, 512],
        "lr_sch_values": [0.00001, 0.00001],
        "lr_sch_transitions": [1000],
        "warmup_steps": 100,
        "lr_end_value": 0.0000001,
        "periods_per_epis": 64,
        "simul_vol_scale": 1.5,
        "init_range": 6,
        "mc_draws": 512,
        "epis_per_step": 32,
        "steps_per_epoch": 100,
        "n_epochs": 100,
        "batch_size": 16,
        "config_eval": {
            "periods_per_epis": 64,
            "mc_draws": 512,
            "simul_vol_scale": 1,
            "eval_n_epis": 32,
            "init_range": 6,
            "proxy_sampler": False,
            "proxy_mcsampler": False,
            "proxy_futurepol": False,
        },
    }
    config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
    config["n_batches"] = config["periods_per_step"] // config["batch_size"]
    return config


def main():
    print("RBC Production Network Model Training - Single Experiment")
    print("=" * 60)

    double_precision = True
    precision = jnp.float64 if double_precision else jnp.float32

    if double_precision:
        jax_config.update("jax_enable_x64", True)

    data_dir = os.path.join(model_dir, "Model_Data")
    save_dir = os.path.join(model_dir, "Experiments/")
    model_name = "Feb21_24_baselinev3.mat"

    model_file = f"RbcProdNet_SolData_{model_name}"
    model_path = os.path.join(data_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Using data directory: {data_dir}")
    print(f"Using experiments directory: {save_dir}")
    print(f"Using model file: {model_file}")

    print("Loading economic model data...")
    model_data = sio.loadmat(model_path, simplify_cells=True)

    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    # ===================================================================
    # CREATE ECONOMIC MODEL
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating economic model...")

    params = model_data["SolData"]["parameters"].copy()
    params["pareps_c"] = 0.5

    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=double_precision,
    )
    print(f"Created model with eps_c = {params['pareps_c']}")

    # ===================================================================
    # CREATE NEURAL NETWORK BUILDER
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating neural network builder...")

    dim_policies = len(model_data["SolData"]["policies_ss"])
    neural_net_builder = create_neural_net_builder(dim_policies=dim_policies, precision=precision)
    print("Configured baseline neural network builder")

    # ===================================================================
    # CREATE EPOCH TRAIN FUNCTION
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating epoch train function...")

    epoch_train_fn = create_fast_epoch_train_fn
    print("Configured baseline epoch train function")

    # ===================================================================
    # CREATE CONFIGURATION
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating training configuration...")

    config = create_config(model_name, save_dir, seed=7)
    config["n_epochs"] = 10
    config["lr_sch_values"] = [0.001, 0.001]

    print("Configuration created:")
    print(f"  Experiment name: {config['exper_name']}")
    print(f"  Seed: {config['seed']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Learning rate: {config['lr_sch_values']}")
    print(f"  Hidden layers: {config['layers']}")

    # ===================================================================
    # CREATE NEURAL NETWORK
    # ===================================================================
    print("\n" + "=" * 60)
    print("Initializing neural network...")

    neural_net = neural_net_builder(config["layers"])
    print("Neural network initialized")

    # ===================================================================
    # RUN EXPERIMENT
    # ===================================================================
    print("\n" + "=" * 60)
    print("Starting training...")

    try:
        result = run_experiment(
            config=config,
            econ_model=econ_model,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
        )
        print("\nExperiment completed successfully!")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback

        traceback.print_exc()
        return None

    # ===================================================================
    # GENERATE PLOTS
    # ===================================================================
    print("\n" + "=" * 60)
    print("Generating training plots...")

    exp_name = config["exper_name"]
    plots_dir = os.path.join(save_dir, exp_name)

    plot_training_metrics(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

    plot_learning_rate_schedule(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

    plot_training_summary(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

    print(f"All plots saved to: {plots_dir}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if "metrics" in result:
        metrics = result["metrics"]
        print(f"\nExperiment: {exp_name}")
        print(f"  Min Loss: {metrics['min_loss']:.6f}")
        print(f"  Max Mean Accuracy: {metrics['max_mean_acc']:.6f}")
        print(f"  Max Min Accuracy: {metrics['max_min_acc']:.6f}")
        print(f"  Time (minutes): {metrics['time_fullexp_minutes']:.2f}")

    print("\nTraining finished!")

    return result


if __name__ == "__main__":
    main()
