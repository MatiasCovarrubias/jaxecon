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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
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
from DEQN.training.run_experiment import run_experiment  # noqa: E402

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "exper_name": "low_volatility",
    "model_dir": "RbcProdNet_reduced",
    # Basic experiment settings
    "date": "Oct4_2025",
    "seed": 1,
    "restore": False,
    "restore_exper_name": None,
    "comment": "",
    # Econ Model parameters
    "model_param_overrides": {
        # "pareps_c": 0.1,
    },
    "mc_draws": 32,  # number of monte-carlo draws for loss calculation
    "init_range": 6,  # range around the SS for state initialization in the model.
    "model_vol_scale": 0.1,  # scale for model volatility (used for simulation and expectation)
    "simul_vol_scale": 10.0,  # scale for simulation volatility (only used in simulation)
    # Training parameters
    "double_precision": True,  # use double precision for the model
    "layers": [32, 32],
    "learning_rate": 0.0005,  # initial learning rate (cosine decay to 0)
    "periods_per_epis": 32,
    "epis_per_step": 32,
    "steps_per_epoch": 100,
    "n_epochs": 100,
    "checkpoint_every_n_epochs": 10,
    # Evaluation configuration
    "config_eval": {
        "periods_per_epis": 64,
        "mc_draws": 64,
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

    # Create reduced policy vectors and matrices
    print("Building reduced policy setup (4N layout: [L, M, Inv, P])...", flush=True)
    N = int(n_sectors)

    # Full policy vector layout: [C, L, Pk, Pm, M, Mout, Inv, Iout, P, Q, Y, Cagg, Lagg, Yagg, Iagg, Magg]
    # Blocks:                     [0, 1, 2,  3,  4, 5,    6,   7,    8, 9, 10, -,    -,    -,    -,    -   ]
    # We want blocks [1, 4, 6, 8] = [L, M, Inv, P]

    # Convert full arrays to numpy first for easier manipulation
    policies_ss_full = jnp.array(model_data["SolData"]["policies_ss"], dtype=precision)
    policies_sd_full = jnp.array(model_data["SolData"]["policies_sd"], dtype=precision)
    C_full = jnp.array(model_data["SolData"]["C"], dtype=precision)

    print(f"Full policies shape: {policies_ss_full.shape}, C shape: {C_full.shape}")
    print(f"Expected: policies={(11*N+5,)}, C={(11*N+5, 2*N)}")

    # EXPLICIT extraction - much clearer than boolean mask
    # Block 1 (L): indices [N:2N]
    # Block 4 (M): indices [4N:5N]
    # Block 6 (Inv): indices [6N:7N]
    # Block 8 (P): indices [8N:9N]

    L_ss = policies_ss_full[N : 2 * N]
    M_ss = policies_ss_full[4 * N : 5 * N]
    Inv_ss = policies_ss_full[6 * N : 7 * N]
    P_ss = policies_ss_full[8 * N : 9 * N]

    L_sd = policies_sd_full[N : 2 * N]
    M_sd = policies_sd_full[4 * N : 5 * N]
    Inv_sd = policies_sd_full[6 * N : 7 * N]
    P_sd = policies_sd_full[8 * N : 9 * N]

    C_L = C_full[N : 2 * N, :]
    C_M = C_full[4 * N : 5 * N, :]
    C_Inv = C_full[6 * N : 7 * N, :]
    C_P = C_full[8 * N : 9 * N, :]

    # Concatenate in the order [L, M, Inv, P]
    policies_ss_reduced = jnp.concatenate([L_ss, M_ss, Inv_ss, P_ss])
    policies_sd_reduced = jnp.concatenate([L_sd, M_sd, Inv_sd, P_sd])
    C_reduced = jnp.concatenate([C_L, C_M, C_Inv, C_P], axis=0)

    print(f"Reduced to: policies={(policies_ss_reduced.shape)}, C={(C_reduced.shape)}")
    print(f"Expected:   policies={(4*N,)}, C={(4*N, 2*N)}")

    # VERIFICATION TEST with dummy state
    print("\n=== VERIFICATION TEST ===")
    dummy_state_logdev = jnp.ones((2 * N,)) * 0.01  # Small log-deviation

    # Full system baseline
    baseline_full_logdev = dummy_state_logdev @ C_full.T
    print(f"Full baseline output shape: {baseline_full_logdev.shape} (expect {(11*N+5,)})")

    # Reduced system baseline
    baseline_reduced_logdev = dummy_state_logdev @ C_reduced.T
    print(f"Reduced baseline output shape: {baseline_reduced_logdev.shape} (expect {(4*N,)})")

    # Check: The reduced baseline should match the corresponding elements from full baseline
    full_L = baseline_full_logdev[N : 2 * N]
    full_M = baseline_full_logdev[4 * N : 5 * N]
    full_Inv = baseline_full_logdev[6 * N : 7 * N]
    full_P = baseline_full_logdev[8 * N : 9 * N]

    reduced_L = baseline_reduced_logdev[0:N]
    reduced_M = baseline_reduced_logdev[N : 2 * N]
    reduced_Inv = baseline_reduced_logdev[2 * N : 3 * N]
    reduced_P = baseline_reduced_logdev[3 * N : 4 * N]

    print("\nBaseline matching test:")
    print(f"  L match:   max diff = {jnp.max(jnp.abs(full_L - reduced_L)):.2e}")
    print(f"  M match:   max diff = {jnp.max(jnp.abs(full_M - reduced_M)):.2e}")
    print(f"  Inv match: max diff = {jnp.max(jnp.abs(full_Inv - reduced_Inv)):.2e}")
    print(f"  P match:   max diff = {jnp.max(jnp.abs(full_P - reduced_P)):.2e}")

    print("\nSteady state values (first 3 of each):")
    print(f"  L_ss:   {L_ss[:3]}")
    print(f"  M_ss:   {M_ss[:3]}")
    print(f"  Inv_ss: {Inv_ss[:3]}")
    print(f"  P_ss:   {P_ss[:3]}")

    # Check if steady states are reasonable (in log space, should be small)
    print("\nSteady state magnitude checks:")
    print(f"  L_ss max abs:   {jnp.max(jnp.abs(L_ss)):.4f}")
    print(f"  M_ss max abs:   {jnp.max(jnp.abs(M_ss)):.4f}")
    print(f"  Inv_ss max abs: {jnp.max(jnp.abs(Inv_ss)):.4f}")
    print(f"  P_ss max abs:   {jnp.max(jnp.abs(P_ss)):.4f}")
    print(f"  policies_sd_reduced min/max: {jnp.min(policies_sd_reduced):.4f} / {jnp.max(policies_sd_reduced):.4f}")

    # Verify model_data file path
    print(f"\nLoaded model_data from: {model_path}")
    print(f"Model directory: {model_dir}")
    print("=== END VERIFICATION ===\n")

    # STEADY STATE TEST - Evaluate loss at exact steady state
    print("\n=== STEADY STATE TEST ===")
    params_original = model_data["SolData"]["parameters"].copy()
    state_sd = jnp.array(model_data["SolData"]["states_sd"], dtype=precision)

    # Create a temporary model for testing
    temp_model = Model(
        parameters=params_original,
        state_ss=state_ss,
        policies_ss=policies_ss_reduced,
        state_sd=state_sd,
        policies_sd=policies_sd_reduced,
        double_precision=config["double_precision"],
        volatility_scale=1.0,
    )

    # Test at exact steady state (normalized to zero)
    state_at_ss = jnp.zeros(2 * N)  # Zero in normalized space = steady state
    policy_at_ss = jnp.zeros(4 * N)  # Zero in normalized space = steady state

    # Compute expectation at SS (should also be at SS)
    shock_zero = jnp.zeros(N)
    state_next_ss = temp_model.step(state_at_ss, policy_at_ss, shock_zero)
    expect_at_ss = temp_model.expect_realization(state_next_ss, policy_at_ss)

    # Evaluate loss at SS
    loss_ss, acc_ss, min_acc_ss, mean_accs_focs_ss, min_accs_focs_ss = temp_model.loss(
        state_at_ss, expect_at_ss, policy_at_ss
    )

    print(f"Loss at steady state: {loss_ss:.6e}")
    print(f"Mean accuracy at SS: {acc_ss:.6f}")
    print(f"Mean accuracies by FOC at SS: {mean_accs_focs_ss}")
    print("  [0]=C_loss, [1]=L_loss, [2]=K_loss, [3]=M_loss")
    print(f"Min accuracies by FOC at SS: {min_accs_focs_ss}")

    # Check if step returns to SS
    print(f"\nState after step at SS (should be ~0): max abs = {jnp.max(jnp.abs(state_next_ss)):.6e}")

    print("=== END STEADY STATE TEST ===\n")

    # Create economic model with reduced policies
    print("Creating economic model...", flush=True)

    params_train = params_original.copy()
    if config["model_param_overrides"] is not None:
        for param_name, param_value in config["model_param_overrides"].items():
            params_train[param_name] = param_value

    econ_model = Model(
        parameters=params_train,
        state_ss=state_ss,
        policies_ss=policies_ss_reduced,
        state_sd=state_sd,
        policies_sd=policies_sd_reduced,
        double_precision=config["double_precision"],
        volatility_scale=config["model_vol_scale"],
    )
    print("Economic model created successfully.", flush=True)

    econ_model_eval = Model(
        parameters=params_original,
        state_ss=state_ss,
        policies_ss=policies_ss_reduced,
        state_sd=state_sd,
        policies_sd=policies_sd_reduced,
        double_precision=config["double_precision"],
        volatility_scale=1.0,
    )
    print("Evaluation model created with original parameters and standard volatility (1.0).", flush=True)

    # Create neural network with reduced setup
    print("Creating neural network...", flush=True)
    dim_policies = 4 * N

    # EXPERIMENT: Test with NO baseline (set C to zeros)
    # This will tell us if the C matrix mismatch is the problem
    USE_BASELINE = False  # Set to False to test without baseline

    if USE_BASELINE:
        C_for_net = C_reduced
        print("Using log-linear baseline from C matrix", flush=True)
    else:
        C_for_net = jnp.zeros_like(C_reduced)
        print("WARNING: Using ZERO baseline (no log-linear approximation)", flush=True)
        print("  This is a test to see if the baseline C matrix is causing the high loss", flush=True)

    neural_net = NeuralNet(
        features=config["layers"] + [dim_policies],
        C=C_for_net,
        state_sd=state_sd,
        policies_sd=policies_sd_reduced,
        param_dtype=precision,
    )
    print("Neural network created successfully.", flush=True)

    # Run training
    print("Starting training...", flush=True)
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
