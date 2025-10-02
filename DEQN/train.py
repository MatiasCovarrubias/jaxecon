#!/usr/bin/env python3
"""
Training script. YOu need to define

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.train

        # Method 2: Run directly as script (from repository root):
        python DEQN/train.py

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

    base_dir = "/content/drive/MyDrive/Jaxecon/DEQN"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import importlib  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.algorithm import create_epoch_train_fn  # noqa: E402
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet  # noqa: E402
from DEQN.training.plots import (
    plot_learning_rate_schedule,
    plot_training_metrics,
    plot_training_summary,
)
from DEQN.training.run_experiment import run_experiment_orbax  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "exper_name": "test_local_1thread",
    "model_dir": "RbcProdNet_Oct2025",
    # Basic experiment settings
    "date": "Oct1_2025",
    "seed": 1,
    "restore": False,
    "restore_exper_name": None,
    "comment": "We now have multit-threading with 4 cores. Also, we use another seed.",
    # Econ Model parameters
    "model_param_overrides": {
        "pareps_c": 0.5,
    },
    "mc_draws": 16,  # number of monte-carlo draws for loss calculation
    "init_range": 6,  # range around the SS for state initialization in the model.
    "model_vol_scale": 1.0,  # scale for model volatility (used for simulation and expectation)
    "simul_vol_scale": 1.0,  # scale for simulation volatility (only used in simulation)
    # Training parameters
    "double_precision": True,  # use double precision for the model
    "layers": [64, 64],
    "learning_rate": 0.001,  # initial learning rate (cosine decay to 0)
    "periods_per_epis": 16,
    "epis_per_step": 64,
    "steps_per_epoch": 100,
    "n_epochs": 10,
    "checkpoint_frequency": 10,
    # Evaluation configuration
    "config_eval": {
        "periods_per_epis": 64,
        "mc_draws": 128,
        "simul_vol_scale": 1,
        "eval_n_epis": 64,
        "init_range": 6,
    },
}

# Derived settings
config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
config["batch_size"] = 16  # hard coded
config["n_batches"] = config["periods_per_step"] // 16

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    print(f"Training: {config['exper_name']}", flush=True)

    # Environment and precision setup
    print("Setting up precision...", flush=True)
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)
    print("Precision setup complete.", flush=True)

    model_dir = os.path.join(base_dir, config["model_dir"])
    save_dir = os.path.join(model_dir, "experiments/")
    config["save_dir"] = save_dir

    # Load model data
    print("Loading model data...", flush=True)
    model_path = os.path.join(model_dir, "model_data.mat")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    print("Model data loaded successfully.", flush=True)
    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    # Create economic model
    print("Creating economic model...", flush=True)
    params = model_data["SolData"]["parameters"].copy()
    state_sd = model_data["SolData"]["states_sd"]
    policies_sd = model_data["SolData"]["policies_sd"]

    if config["model_param_overrides"] is not None:
        for param_name, param_value in config["model_param_overrides"].items():
            params[param_name] = param_value

    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
        volatility_scale=config["model_vol_scale"],
    )
    print("Economic model created successfully.", flush=True)

    # Create neural network
    print("Creating neural network...", flush=True)
    dim_policies = len(model_data["SolData"]["policies_ss"])
    neural_net = NeuralNet(
        features=config["layers"] + [dim_policies],
        C=model_data["SolData"]["C"],
        policies_sd=policies_sd,
        param_dtype=precision,
    )
    print("Neural network created successfully.", flush=True)

    # Run training
    print("Starting training...", flush=True)
    epoch_train_fn = create_epoch_train_fn

    try:
        result = run_experiment_orbax(
            config=config,
            econ_model=econ_model,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Generate plots
    if result:
        exp_name = config["exper_name"]
        plots_dir = os.path.join(config["save_dir"], exp_name)

        plot_training_metrics(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)
        plot_learning_rate_schedule(
            training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100
        )
        plot_training_summary(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

        if "metrics" in result:
            m = result["metrics"]
            print(f"Loss: {m['min_loss']:.7f} | Acc: {m['max_mean_acc']:.4f} | Time: {m['time_fullexp_minutes']:.1f}m")

    return result


if __name__ == "__main__":
    main()
