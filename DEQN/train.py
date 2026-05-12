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

    base_dir = "/content/drive/MyDrive/Jaxecontemp"

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
from DEQN.analysis.model_hooks import load_model_analysis_hooks  # noqa: E402
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet  # noqa: E402
from DEQN.training.run_experiment import load_experiment_train_state, run_experiment  # noqa: E402
from DEQN.training.utils_train import (  # noqa: E402
    _build_ir_finetune_config,
    _get_ir_finetune_source_exper_name,
    _is_ir_finetune_enabled,
    _plot_result,
    _print_metrics_summary,
    _resolve_model_data_file,
    _set_derived_training_config,
)

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "exper_name": "benchmark_final_long",
    "model_dir": "RbcProdNet_April2026",
    # MATLAB data file and object name. Set model_data_file to None to use defaults.
    "model_data_file": "ModelData_newwds_v2.mat",
    "model_data_object": "ModelData",
    # Basic experiment settings
    "date": "May7_2026",
    "seed": 1,
    "restore": False,
    "restore_exper_name": "",
    "restore_step": False,  # If True, continue from checkpoint's step count (low LR). If False, reset to step 0.
    "comment": "",
    # Econ Model parameters
    "model_param_overrides": {
        # "pareps_c": 0.1,
    },
    "mc_draws": 128,  # number of monte-carlo draws for loss calculation
    "init_range": 5,  # range around SS (% deviation). Can be scalar or dict.
    "model_vol_scale": 1.0,  # scale for model volatility (used for simulation and expectation)
    "simul_vol_scale": 1.0,  # scale for simulation volatility (only used in simulation)
    # Training parameters
    "double_precision": True,  # use double precision for the model
    "layers": [256, 256],
    "learning_rate": 0.0005,  # initial learning rate (cosine decay to 0)
    "periods_per_epis": 64,
    "epis_per_step": 64,
    "steps_per_epoch": 100,
    "n_epochs": 0,
    "checkpoint_every_n_epochs": 10,
    # Optional IR fine-tuning stage. Shock sizes are percentages.
    "config_ir_finetune": {
        "enabled": True,
        "source_exper_name": "benchmark_final_long",
        "source_step": None,
        "exper_name": "benchmark_final_long_IR",
        "exper_suffix": "_IR",
        "min_shock_size": 5.0,
        "max_shock_size": 30.0,
        "learning_rate": 0.00005,
        "n_epochs": 1000,
        "eval_ir_rollouts": True,
    },
    # Evaluation configuration
    "config_eval": {
        "periods_per_epis": 128,
        "mc_draws": 256,
        "simul_vol_scale": 1.0,
        "eval_n_epis": 128,
        "init_range": 5,
    },
}

# Derived settings
_set_derived_training_config(config)

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model
analysis_hooks = load_model_analysis_hooks(config["model_dir"])


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

    # Load model data (supports both old and new structure)
    print("Loading model data...", flush=True)
    model_data_file, model_path = _resolve_model_data_file(
        model_dir,
        config.get("model_data_file"),
        ["ModelData.mat", "model_data.mat"],
    )
    config["model_data_file"] = model_data_file
    model_data = sio.loadmat(model_path, simplify_cells=True)
    print(f"Model data loaded successfully from {model_data_file}.", flush=True)

    # Detect structure and extract data
    model_data_object = config.get("model_data_object", "ModelData")
    if model_data_object in model_data:
        # New structure (Dec 2025+): ModelData.SteadyState, ModelData.Statistics, ModelData.Solution
        print(f"Detected new {model_data_object} structure.", flush=True)
        md = model_data[model_data_object]

        # Extract from SteadyState
        ss = md["SteadyState"]
        n_sectors = ss["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(ss["endostates_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params_original = ss["parameters"].copy()
        policies_ss_raw = jnp.array(ss["policies_ss"], dtype=precision)

        # Extract from Statistics (NOT Simulation!)
        stats = md["Statistics"]
        state_sd = jnp.array(stats["states_sd"], dtype=precision)
        policies_sd_raw = jnp.array(stats["policies_sd"], dtype=precision)

        # Extract from Solution
        C_matrix = md["Solution"]["StateSpace"]["C"]

        # Handle potential size mismatch between policies_ss and policies_sd
        # policies_ss may include V (value function) at the end which isn't in policies_sd
        n_policies = min(len(policies_sd_raw), len(policies_ss_raw))
        if len(policies_sd_raw) != len(policies_ss_raw):
            print(
                f"  Note: Aligning policy dimensions ({len(policies_ss_raw)} ss, {len(policies_sd_raw)} sd) -> {n_policies}",
                flush=True,
            )
        policies_ss = policies_ss_raw[:n_policies]
        policies_sd = policies_sd_raw[:n_policies]

    elif "SolData" in model_data:
        # Old structure: SolData contains everything
        print("Detected old SolData structure.", flush=True)
        soldata = model_data["SolData"]
        n_sectors = soldata["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(soldata["k_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params_original = soldata["parameters"].copy()
        state_sd = jnp.array(soldata["states_sd"], dtype=precision)
        policies_sd = jnp.array(soldata["policies_sd"], dtype=precision)
        policies_ss = jnp.array(soldata["policies_ss"], dtype=precision)
        C_matrix = soldata["C"]

    else:
        available_keys = sorted(key for key in model_data.keys() if not key.startswith("__"))
        raise ValueError(
            "Unknown model_data structure. "
            f"Expected '{model_data_object}' or 'SolData' key. Available keys: {available_keys}"
        )

    print(f"  n_sectors: {n_sectors}", flush=True)
    print(f"  state_ss shape: {state_ss.shape}", flush=True)
    print(f"  policies_ss shape: {policies_ss.shape}", flush=True)
    print(f"  policies_sd shape: {policies_sd.shape}", flush=True)

    params_train = params_original.copy()
    if config["model_param_overrides"] is not None:
        for param_name, param_value in config["model_param_overrides"].items():
            params_train[param_name] = param_value

    econ_model = Model(
        parameters=params_train,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
        volatility_scale=config["model_vol_scale"],
    )
    print("Economic model created successfully.", flush=True)

    econ_model_eval = Model(
        parameters=params_original,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
        volatility_scale=1.0,
    )
    print("Evaluation model created with original parameters and standard volatility (1.0).", flush=True)

    # Create neural network
    print("Creating neural network...", flush=True)
    dim_policies = econ_model.dim_policies
    neural_net = NeuralNet(
        features=config["layers"] + [dim_policies],
        C=C_matrix,
        states_sd=state_sd,
        policies_sd=policies_sd,
        param_dtype=precision,
    )
    print("Neural network created successfully.", flush=True)

    if _is_ir_finetune_enabled(config) and config["n_epochs"] == 0:
        if not _get_ir_finetune_source_exper_name(config) and not config.get("restore", False):
            raise ValueError(
                "IR fine-tuning with baseline n_epochs=0 requires either "
                "config_ir_finetune['source_exper_name'] or top-level restore=True."
            )

    # Run training
    print("Starting baseline training stage...", flush=True)
    epoch_train_fn = create_epoch_train_fn

    try:
        result = run_experiment(
            config=config,
            econ_model=econ_model,
            neural_net=neural_net,
            epoch_train_fn=epoch_train_fn,
            econ_model_eval=econ_model_eval,
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

        _plot_result(result, plots_dir, exp_name)

        if "metrics" in result:
            _print_metrics_summary("Baseline", result["metrics"])
            if result["metrics"].get("training_skipped"):
                print("Baseline training skipped; no baseline checkpoint or plots written.", flush=True)
            else:
                print(f"Baseline experiment stored in: {plots_dir}", flush=True)

    if result and _is_ir_finetune_enabled(config):
        from DEQN.algorithm import create_ir_finetune_epoch_train_fn  # noqa: E402

        ir_config = _build_ir_finetune_config(config, econ_model, analysis_hooks)
        ir_exp_name = ir_config["exper_name"]
        ir_plots_dir = os.path.join(ir_config["save_dir"], ir_exp_name)
        print(
            f"Starting IR fine-tuning: {ir_exp_name} "
            f"({ir_config['ir_finetune_min_shock_size']}%-{ir_config['ir_finetune_max_shock_size']}% shocks)",
            flush=True,
        )

        try:
            ir_initial_params = result["train_state"].params
            source_exper_name = _get_ir_finetune_source_exper_name(config)
            if source_exper_name:
                source_step = (config.get("config_ir_finetune") or {}).get("source_step")
                print(
                    f"Loading IR fine-tuning source checkpoint: {source_exper_name}"
                    + (f" at step {source_step}" if source_step is not None else ""),
                    flush=True,
                )
                source_state = load_experiment_train_state(
                    config=ir_config,
                    econ_model=econ_model,
                    neural_net=neural_net,
                    experiment_name=source_exper_name,
                    step=source_step,
                    restore_step=False,
                )
                ir_initial_params = source_state.params

            ir_result = run_experiment(
                config=ir_config,
                econ_model=econ_model,
                neural_net=neural_net,
                epoch_train_fn=create_ir_finetune_epoch_train_fn,
                econ_model_eval=econ_model_eval,
                initial_params=ir_initial_params,
            )
        except Exception as e:
            print(f"IR fine-tuning failed: {e}")
            import traceback

            traceback.print_exc()
            os.makedirs(ir_plots_dir, exist_ok=True)
            with open(os.path.join(ir_plots_dir, "ir_finetune_error.txt"), "w") as error_file:
                error_file.write(traceback.format_exc())
            print("Baseline experiment remains saved; returning baseline result.", flush=True)
            return result

        try:
            _plot_result(ir_result, ir_plots_dir, ir_exp_name)
            if "metrics" in ir_result:
                _print_metrics_summary("IR", ir_result["metrics"])
            result["ir_finetune_result"] = ir_result
        except Exception as e:
            print(f"IR fine-tuning completed, but IR plotting failed: {e}")
            import traceback

            traceback.print_exc()
            os.makedirs(ir_plots_dir, exist_ok=True)
            with open(os.path.join(ir_plots_dir, "ir_finetune_plot_error.txt"), "w") as error_file:
                error_file.write(traceback.format_exc())
            result["ir_finetune_result"] = ir_result

    return result


if __name__ == "__main__":
    main()
