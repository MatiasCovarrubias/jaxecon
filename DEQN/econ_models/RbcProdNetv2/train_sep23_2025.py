#!/usr/bin/env python3
"""
RBC ProdNet Model Training Script

This script provides a training framework for neural network policies
for the RBC Production Network model using the DEQN solver. It supports
running multiple experiments with different configurations.

The script uses a simple approach: create lists of configs, models, builders,
and training functions, then `generate_experiment_grid` creates all combinations.

Example:
    # Create configs manually
    config1 = create_base_config(..., seed=7)
    config2 = create_base_config(..., seed=42)

    # Create lists
    configs = [config1, config2]
    econ_models = [model_low_ies, model_high_ies]
    neural_net_builders = [baseline_builder]
    epoch_train_functions = [baseline_train_fn]

    # Generate grid (2 configs × 2 models × 1 builder × 1 train_fn = 4 experiments)
    experiments = generate_experiment_grid(
        configs=configs,
        econ_models=econ_models,
        neural_net_builders=neural_net_builders,
        epoch_train_functions=epoch_train_functions
    )

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.econ_models.RbcProdNetv2.train_sep23_2025

        # Method 2: Run directly as script (from repository root):
        python DEQN/econ_models/RbcProdNetv2/train_sep23_2025.py

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

# Automatically detect if we're running in Google Colab
try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    # ========================================================================
    # COLAB SETUP
    # ========================================================================

    # Install JAX with CUDA support
    print("Installing JAX with CUDA support...")
    import subprocess

    subprocess.run(["pip", "install", "--upgrade", "jax[cuda12]"], check=True)

    # Clone repository
    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    # Add to Python path
    sys.path.insert(0, "/content/jaxecon")

    # Mount Google Drive
    print("Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    # Set model directory to Google Drive location
    model_dir = "/content/drive/MyDrive/Jaxecon/RbcProdNet/"

else:
    # ========================================================================
    # LOCAL SETUP
    # ========================================================================

    # Add repository root to path for absolute imports when run directly
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    model_dir = os.path.join(repo_root, "DEQN", "econ_models", "RbcProdNetv2")

# ============================================================================
# IMPORTS (same for both environments)
# ============================================================================

import jax.numpy as jnp  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

# DEQN imports (use absolute imports that work both as module and script)
from DEQN.algorithm import (  # noqa: E402
    create_epoch_train_fn,
    create_fast_epoch_train_fn,
)
from DEQN.econ_models.RbcProdNetv2.plots import (  # noqa: E402
    plot_learning_rate_schedule,
    plot_training_metrics,
    plot_training_summary,
)
from DEQN.econ_models.RbcProdNetv2.RbcProdNet_Sept23_2025 import Model  # noqa: E402
from DEQN.neural_nets.neural_nets import create_neural_net_builder  # noqa: E402
from DEQN.neural_nets.with_loglinear_baseline import (  # noqa: E402
    create_neural_net_loglinear_builder,
)
from DEQN.training.runner import generate_experiment_grid, run_experiment  # noqa: E402

# Configure JAX debugging
jax_config.update("jax_debug_nans", True)


def create_base_config(model_name, save_dir, exper_name="baseline_experiment", seed=7):
    """Create base configuration for training experiments."""
    config = {
        "model_name": model_name,
        "exper_name": exper_name,
        "date": "Sep25_2025",
        "save_dir": save_dir,
        "restore": False,
        "restore_exper_name": None,
        "seed": seed,
        "comment": "Baseline experiment configuration.",
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
    print("RBC Production Network Model Training")
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
    # DEFINE ECONOMIC MODELS (AS A LIST)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating economic models...")

    params_low_ies = model_data["SolData"]["parameters"].copy()
    params_low_ies["pareps_c"] = 0.5
    model_low_ies = Model(
        parameters=params_low_ies,
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=double_precision,
    )
    print(f"Created model with eps_c = {params_low_ies['pareps_c']}")

    params_high_ies = model_data["SolData"]["parameters"].copy()
    params_high_ies["pareps_c"] = 0.99
    model_high_ies = Model(
        parameters=params_high_ies,
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=double_precision,
    )
    print(f"Created model with eps_c = {params_high_ies['pareps_c']}")

    econ_models = [model_low_ies, model_high_ies]

    # ===================================================================
    # DEFINE NEURAL NETWORK BUILDERS (AS A LIST)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Defining neural network builders...")

    C = model_data["SolData"]["C"]
    policies_sd = jnp.array(model_data["SolData"]["policies_sd"], dtype=precision)
    dim_policies = len(model_data["SolData"]["policies_ss"])

    baseline_builder = create_neural_net_builder(dim_policies=dim_policies, precision=precision)
    loglinear_builder = create_neural_net_loglinear_builder(
        C=C,
        policies_sd=policies_sd,
        dim_policies=dim_policies,
        param_dtype=precision,
    )

    neural_net_builders = [baseline_builder, loglinear_builder]
    print("Configured baseline and loglinear neural network builders")

    # ===================================================================
    # DEFINE EPOCH TRAIN FUNCTIONS (AS A LIST)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Defining epoch train functions...")

    epoch_train_functions = [create_epoch_train_fn, create_fast_epoch_train_fn]
    print("Configured baseline and fast epoch train functions")

    # ===================================================================
    # CREATE CONFIGS (MANUALLY COPY AND MODIFY AS NEEDED)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Creating configuration variants...")

    config_seed7 = create_base_config(model_name, save_dir, seed=7)
    config_seed7["n_epochs"] = 10
    config_seed7["lr_sch_values"] = [0.001, 0.001]

    config_seed42 = create_base_config(model_name, save_dir, seed=42)
    config_seed42["n_epochs"] = 10
    config_seed42["lr_sch_values"] = [0.0001, 0.0001]

    configs = [config_seed7, config_seed42]
    print(f"Created {len(configs)} configuration variants")

    # ===================================================================
    # GENERATE EXPERIMENT GRID
    # ===================================================================
    print("\n" + "=" * 60)
    print("Generating experiment grid...")
    print(f"  Configs: {len(configs)}")
    print(f"  Economic models: {len(econ_models)}")
    print(f"  Neural net builders: {len(neural_net_builders)}")
    print(f"  Epoch train functions: {len(epoch_train_functions)}")

    total_experiments = len(configs) * len(econ_models) * len(neural_net_builders) * len(epoch_train_functions)
    print(f"  Total experiments: {total_experiments}")

    experiments_to_run = generate_experiment_grid(
        configs=configs,
        econ_models=econ_models,
        neural_net_builders=neural_net_builders,
        epoch_train_functions=epoch_train_functions,
    )

    print(f"Generated {len(experiments_to_run)} experiments")

    # ===================================================================
    # RUN EXPERIMENTS AND COLLECT DATA
    # ===================================================================
    print("\n" + "=" * 60)
    print("Starting experiments...")

    training_results = {}

    for i, exp_spec in enumerate(experiments_to_run, 1):
        print("\n" + "=" * 60)
        print(f"EXPERIMENT {i}/{len(experiments_to_run)}: {exp_spec['name']}")
        print("=" * 60)

        econ_model = exp_spec["econ_model"]
        nn_builder = exp_spec["neural_net_builder"]
        epoch_train_fn = exp_spec["epoch_train_fn"]
        config = exp_spec["config"]

        neural_net = nn_builder(config["layers"])

        try:
            result = run_experiment(
                config=config,
                econ_model=econ_model,
                neural_net=neural_net,
                epoch_train_fn=epoch_train_fn,
            )
            training_results[exp_spec["name"]] = result
            print(f"\nExperiment {exp_spec['name']} completed successfully!")
        except Exception as e:
            print(f"\nExperiment {exp_spec['name']} failed with error: {e}")
            import traceback

            traceback.print_exc()
            training_results[exp_spec["name"]] = {"status": "failed", "error": str(e)}

    # ===================================================================
    # SAVE EXPERIMENT SUMMARY TO CSV
    # ===================================================================
    print("\n" + "=" * 60)
    print("Saving experiment summary to CSV...")

    csv_filename = os.path.join(model_dir, "experiment_results.csv")
    csv_data = []

    for exp_name, result in training_results.items():
        if "metrics" in result:
            csv_data.append(result["metrics"])

    if csv_data:
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            df = pd.concat([df, pd.DataFrame(csv_data)], ignore_index=True)
        else:
            df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to: {csv_filename}")

    # ===================================================================
    # GENERATE PLOTS FOR EACH EXPERIMENT
    # ===================================================================
    print("\n" + "=" * 60)
    print("Generating training plots...")

    plots_dir = os.path.join(model_dir, "Training_Plots")
    os.makedirs(plots_dir, exist_ok=True)

    for exp_name, result in training_results.items():
        if "metrics" not in result:
            print(f"Skipping plots for {exp_name} (experiment failed)")
            continue

        print(f"\nGenerating plots for {exp_name}...")

        # Generate training metrics plot (losses and accuracies)
        plot_training_metrics(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

        # Generate learning rate schedule plot
        plot_learning_rate_schedule(
            training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100
        )

        # Generate comprehensive summary plot
        plot_training_summary(training_results=result, save_dir=plots_dir, experiment_name=exp_name, display_dpi=100)

    print(f"\nAll plots saved to: {plots_dir}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for exp_name, result in training_results.items():
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"\n{exp_name}:")
            print(f"  Min Loss: {metrics['min_loss']:.6f}")
            print(f"  Max Mean Accuracy: {metrics['max_mean_acc']:.6f}")
            print(f"  Max Min Accuracy: {metrics['max_min_acc']:.6f}")
            print(f"  Time (minutes): {metrics['time_fullexp_minutes']:.2f}")
        else:
            status = result.get("status", "unknown")
            print(f"\n{exp_name}: {status}")

    completed = sum(1 for r in training_results.values() if "metrics" in r)
    print(f"\nCompleted: {completed}/{len(experiments_to_run)}")

    print("\nAll experiments and analysis finished!")

    return training_results


if __name__ == "__main__":
    main()
