#!/usr/bin/env python3
"""
Training script for the Three-Equation New Keynesian model with DEQN.

Based on Chapter 3 of Galí's "Monetary Policy, Inflation, and the Business Cycle".

Usage:
    LOCAL:
        python -m DEQN.econ_models.NK.train
        # or
        python DEQN/econ_models/NK/train.py

    COLAB:
        Copy contents into a cell and run.
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
    base_dir = "/content/drive/MyDrive/Jaxecon/DEQN/econ_models"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402

from DEQN.algorithm.epoch_train import create_epoch_train_fn  # noqa: E402
from DEQN.algorithm.eval import create_eval_fn  # noqa: E402
from DEQN.algorithm.simulation import create_episode_simul_fn  # noqa: E402
from DEQN.econ_models.NK.model import Model  # noqa: E402
from DEQN.neural_nets.neural_nets import NeuralNet  # noqa: E402

jax_config.update("jax_debug_nans", True)

# ============================================================================
# CONFIGURATION
# ============================================================================

config = {
    "exper_name": "nk_baseline",
    "seed": 42,
    # Training parameters
    "double_precision": False,
    "layers": [32, 32],
    "learning_rate": 0.001,
    "mc_draws": 16,
    "init_range": 5,
    "simul_vol_scale": 1.0,
    # Episode structure
    "periods_per_epis": 32,
    "epis_per_step": 16,
    "steps_per_epoch": 50,
    "n_epochs": 100,  # NK model may need more epochs to converge
    # Evaluation config
    "config_eval": {
        "periods_per_epis": 64,
        "mc_draws": 32,
        "simul_vol_scale": 1.0,
        "eval_n_epis": 32,
        "init_range": 5,
    },
}

# Derived settings
config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
config["batch_size"] = 16
config["n_batches"] = config["periods_per_step"] // config["batch_size"]

# ============================================================================
# MAIN
# ============================================================================


def main():
    import jax
    import optax
    from flax.training import train_state
    from time import time

    print(f"\n{'='*60}")
    print("DEQN Training: Three-Equation New Keynesian Model (Galí Ch.3)")
    print(f"{'='*60}\n")

    # Precision setup
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Create economic model
    print("Creating economic model...")
    econ_model = Model(
        precision=precision,
        double_precision=config["double_precision"],
    )
    print(f"  State dim: {econ_model.dim_states} (productivity shock a_t, monetary shock v_t)")
    print(f"  Policy dim: {econ_model.dim_policies} (output gap ỹ_t, inflation π_t)")
    print("\nCalibration (Galí notation):")
    print(f"  β (discount):     {float(econ_model.beta):.4f}")
    print(f"  σ (CRRA):         {float(econ_model.sigma):.4f}")
    print(f"  κ (NKPC slope):   {float(econ_model.kappa):.4f}")
    print(f"  φ_π (Taylor π):   {float(econ_model.phi_pi):.4f}")
    print(f"  φ_y (Taylor ỹ):   {float(econ_model.phi_y):.4f}")
    print(f"  ρ_a (persist a):  {float(econ_model.rho_a):.4f}")
    print(f"  ρ_v (persist v):  {float(econ_model.rho_v):.4f}")

    # Create neural network
    print("\nCreating neural network...")
    features = config["layers"] + [econ_model.dim_policies]
    neural_net = NeuralNet(features=features, precision=precision)
    print(f"  Architecture: {features}")

    # Initialize
    rng = random.PRNGKey(config["seed"])
    rng, rng_init, rng_epoch, rng_eval = random.split(rng, 4)

    params = neural_net.init(rng_init, jnp.zeros_like(econ_model.state_ss))

    # Learning rate schedule
    total_steps = config["n_epochs"] * config["steps_per_epoch"]
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config["learning_rate"],
        decay_steps=total_steps,
        alpha=0.01,
    )

    # Create train state
    train_state_obj = train_state.TrainState.create(
        apply_fn=neural_net.apply,
        params=params,
        tx=optax.adam(lr_schedule),
    )

    # Create training and eval functions
    print("\nCompiling training functions...")
    epoch_train_fn = jax.jit(create_epoch_train_fn(econ_model, config))
    eval_fn = jax.jit(create_eval_fn(econ_model, config))

    # Warmup compilation
    t0 = time()
    _ = epoch_train_fn(train_state_obj, rng_epoch)
    _ = eval_fn(train_state_obj, rng_eval)
    print(f"  Compilation time: {time() - t0:.2f}s")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} {'Loss':>12} {'Mean Acc':>12} {'Min Acc':>12} {'LR':>12}")
    print("-" * 60)

    t_start = time()
    losses, accuracies = [], []

    for epoch in range(1, config["n_epochs"] + 1):
        # Training step
        train_state_obj, rng_epoch, epoch_metrics = epoch_train_fn(train_state_obj, rng_epoch)

        # Evaluation
        if epoch % 5 == 0 or epoch == 1:
            eval_metrics = eval_fn(train_state_obj, rng_eval)
            mean_loss = float(eval_metrics[0])
            mean_acc = float(eval_metrics[1])
            min_acc = float(eval_metrics[2])
            current_lr = float(lr_schedule(train_state_obj.step))

            losses.append(mean_loss)
            accuracies.append(mean_acc)

            print(f"{epoch:>6} {mean_loss:>12.6f} {mean_acc:>12.4f} {min_acc:>12.4f} {current_lr:>12.6f}")

    total_time = time() - t_start
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final accuracy: {accuracies[-1]:.4f}")

    # Run a simulation to show results
    print(f"\n{'='*60}")
    print("Running Simulation")
    print(f"{'='*60}")

    simul_fn = create_episode_simul_fn(econ_model, {"periods_per_epis": 100, "init_range": 0, "simul_vol_scale": 1.0})
    simul_obs = simul_fn(train_state_obj, rng)
    simul_policies = train_state_obj.apply_fn(train_state_obj.params, simul_obs)

    aggregates = econ_model.get_aggregates(simul_policies, simul_obs)

    print("\nSimulation Statistics (100 periods from steady state):")
    print(f"  {'Variable':<12} {'Mean':>12} {'Std':>12}")
    print("  " + "-" * 36)
    var_labels = {
        "y_gap": "ỹ (gap)",
        "pi": "π (infl)",
        "i": "i (nom rate)",
        "r": "r (real rate)",
        "r_n": "r^n (nat rate)",
        "a": "a (prod)",
        "v": "v (mon pol)",
    }
    for name, values in aggregates.items():
        label = var_labels.get(name, name)
        print(f"  {label:<12} {float(jnp.mean(values)):>12.6f} {float(jnp.std(values)):>12.6f}")

    # Show correlations
    print("\nKey Correlations:")
    y_gap = aggregates["y_gap"]
    pi = aggregates["pi"]
    i = aggregates["i"]
    v = aggregates["v"]
    a = aggregates["a"]

    corr_y_pi = jnp.corrcoef(y_gap, pi)[0, 1]
    corr_y_i = jnp.corrcoef(y_gap, i)[0, 1]
    corr_pi_i = jnp.corrcoef(pi, i)[0, 1]
    corr_y_a = jnp.corrcoef(y_gap, a)[0, 1]
    corr_y_v = jnp.corrcoef(y_gap, v)[0, 1]

    print(f"  Corr(ỹ, π):  {float(corr_y_pi):>8.4f}")
    print(f"  Corr(ỹ, i):  {float(corr_y_i):>8.4f}")
    print(f"  Corr(π, i):  {float(corr_pi_i):>8.4f}")
    print(f"  Corr(ỹ, a):  {float(corr_y_a):>8.4f}")
    print(f"  Corr(ỹ, v):  {float(corr_y_v):>8.4f}")

    return train_state_obj, losses, accuracies


if __name__ == "__main__":
    main()
