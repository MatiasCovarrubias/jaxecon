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
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet  # noqa: E402
from DEQN.training.runner import run_experiment  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION - All settings defined upfront
# ============================================================================

# Model and experiment names - defined at the very beginning
MODEL_NAME = "Feb21_24_baselinev3.mat"
EXPER_NAME = "baseline_single"
SEED = 7

# Configuration dictionary - everything in one place
config = {
    # Basic experiment settings
    "model_name": MODEL_NAME,
    "exper_name": EXPER_NAME,
    "date": "Sep25_2025",
    "seed": SEED,
    "restore": False,
    "restore_exper_name": None,
    "comment": "Single experiment baseline configuration.",
    "comment_at_end": False,
    # Precision settings
    "double_precision": True,
    # Model parameter overrides - only specify parameters you want to change
    "model_param_overrides": {
        "pareps_c": 0.5,  # Consumption elasticity parameter
        # Add other parameter overrides here as needed
        # "parbeta": 0.99,  # Discount factor
        # "paralpha": 0.33,  # Capital share
    },
    # Model volatility and scaling
    "model_vol_scale": 1.0,  # Scale for model volatility (used for simulation and expectation)
    "simul_vol_scale": 1.5,  # Scale for simulation volatility (only used in simulation)
    # Neural network, learning rate schedule, and batch size
    "layers": [128, 128],
    "lr_sch_values": [0.002, 0.002],
    "lr_sch_transitions": [1000],
    "warmup_steps": 100,
    "lr_end_value": 0.0000001,
    "batch_size": 16,
    # Training parameters
    "init_range": 6,
    "mc_draws": 512,
    "periods_per_epis": 64,
    "epis_per_step": 32,
    "steps_per_epoch": 100,
    "n_epochs": 10,  # Quick test
    # Evaluation configuration
    "config_eval": {
        "periods_per_epis": 64,
        "mc_draws": 512,
        "simul_vol_scale": 1,  # Different from training simul_vol_scale
        "eval_n_epis": 32,
        "init_range": 6,
    },
}

# Derived settings
config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
config["n_batches"] = config["periods_per_step"] // config["batch_size"]
config["precision"] = jnp.float64 if config["double_precision"] else jnp.float32

print("RBC ProdNet Training")
print(f"Config: {config['exper_name']} | {config['n_epochs']} epochs | LR {config['lr_sch_values'][0]}")


def setup_environment_and_load_data():
    """Setup environment and load model data - combined function."""
    # Set precision based on config
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Directory setup
    data_dir = os.path.join(model_dir, "Model_Data")
    save_dir = os.path.join(model_dir, "Experiments/")
    config["save_dir"] = save_dir

    # Load model data
    model_file = f"RbcProdNet_SolData_{config['model_name']}"
    model_path = os.path.join(data_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = sio.loadmat(model_path, simplify_cells=True)

    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=config["precision"])
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    print(f"Loaded: {n_sectors} sectors")
    return model_data, state_ss


def create_economic_model(model_data, state_ss):
    """Create economic model instance using config parameter overrides and volatility scaling."""
    # Start with original parameters
    params = model_data["SolData"]["parameters"].copy()

    # Apply parameter overrides from config
    for param_name, param_value in config["model_param_overrides"].items():
        params[param_name] = param_value
        print(f"Override: {param_name} = {param_value}")

    # Apply volatility scaling to standard deviations
    vol_scale = config["model_vol_scale"]
    state_sd = model_data["SolData"]["states_sd"] * vol_scale
    policies_sd = model_data["SolData"]["policies_sd"] * vol_scale

    if vol_scale != 1.0:
        print(f"Scaled volatility by {vol_scale}")

    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
    )

    return econ_model


def create_neural_network(model_data):
    """Create neural network directly - uses same volatility scaling as economic model."""
    dim_policies = len(model_data["SolData"]["policies_ss"])

    # Use same volatility scaling as in economic model for consistency
    policies_sd = model_data["SolData"]["policies_sd"] * config["model_vol_scale"]

    neural_net = NeuralNet(
        features=config["layers"] + [dim_policies],
        C=model_data["SolData"]["loglin_coeff"],
        policies_sd=policies_sd,
        param_dtype=config["precision"],
    )
    print(f"Network: {config['layers']} -> {dim_policies}")

    return neural_net


def run_training(econ_model, neural_net):
    """Run the training experiment."""
    print("Training...")

    epoch_train_fn = create_fast_epoch_train_fn

    try:
        result = run_experiment(
            config=config,
            econ_model=econ_model,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
        )
        print("✓ Complete")
        return result
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_plots(result):
    """Generate training plots."""
    exp_name = config["exper_name"]
    plots_dir = os.path.join(config["save_dir"], exp_name)

    plot_training_metrics(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)
    plot_learning_rate_schedule(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)
    plot_training_summary(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

    print(f"Plots: {plots_dir}")


def print_summary(result):
    """Print training summary."""
    if result and "metrics" in result:
        m = result["metrics"]
        print(
            f"\nResults: Loss {m['min_loss']:.6f} | Acc {m['max_mean_acc']:.4f} | Time {m['time_fullexp_minutes']:.1f}m"
        )
    print("Done.")


def main():
    """Main training pipeline - simple and clean."""
    # 1. Setup environment and load data
    model_data, state_ss = setup_environment_and_load_data()

    # 2. Create economic model
    econ_model = create_economic_model(model_data, state_ss)

    # 3. Create neural network
    neural_net = create_neural_network(model_data)

    # 4. Run training
    result = run_training(econ_model, neural_net)

    # 5. Generate plots and summary
    if result:
        generate_plots(result)
        print_summary(result)

    return result


if __name__ == "__main__":
    main()
