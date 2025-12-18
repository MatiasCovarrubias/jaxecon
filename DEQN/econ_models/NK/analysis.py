#!/usr/bin/env python3
"""
Analysis script for the Three-Equation New Keynesian model.

Computes and plots impulse responses to:
1. Productivity shock (a_t) - affects natural rate of interest
2. Monetary policy shock (v_t) - deviation from Taylor rule

Based on Chapter 3 of Galí's "Monetary Policy, Inflation, and the Business Cycle".

Usage:
    LOCAL:
        python -m DEQN.econ_models.NK.analysis
        # or
        python DEQN/econ_models/NK/analysis.py

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
    base_dir = "/content/drive/MyDrive/Jaxecon/DEQN/econ_models/NK"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models", "NK")

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import optax  # noqa: E402
from flax.training import train_state  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402
from time import time  # noqa: E402

from DEQN.algorithm.epoch_train import create_epoch_train_fn  # noqa: E402
from DEQN.algorithm.eval import create_eval_fn  # noqa: E402
from DEQN.econ_models.NK.model import Model  # noqa: E402
from DEQN.neural_nets.neural_nets import NeuralNet  # noqa: E402

jax_config.update("jax_debug_nans", True)

# ============================================================================
# CONFIGURATION
# ============================================================================

config = {
    "exper_name": "nk_analysis",
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
    "n_epochs": 100,
    # Evaluation config
    "config_eval": {
        "periods_per_epis": 64,
        "mc_draws": 32,
        "simul_vol_scale": 1.0,
        "eval_n_epis": 32,
        "init_range": 5,
    },
    # Impulse response config
    "ir_periods": 40,  # Number of periods for IRFs
    "shock_size": 1.0,  # Size of shock (in std deviations of innovation)
}

# Derived settings
config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
config["batch_size"] = 16
config["n_batches"] = config["periods_per_step"] // config["batch_size"]


# ============================================================================
# IMPULSE RESPONSE FUNCTIONS
# ============================================================================


def compute_impulse_response(train_state_obj, econ_model, shock_idx, shock_size, n_periods):
    """
    Compute impulse response to a one-time shock.

    Args:
        train_state_obj: Trained model state
        econ_model: Economic model instance
        shock_idx: Index of shock (0=productivity, 1=monetary policy)
        shock_size: Size of initial shock (in innovation units)
        n_periods: Number of periods to simulate

    Returns:
        Dictionary of impulse responses for each variable
    """
    # Initialize at steady state
    state = jnp.zeros(econ_model.dim_states, dtype=econ_model.precision)

    # Create initial shock
    initial_shock = jnp.zeros(2, dtype=econ_model.precision)
    initial_shock = initial_shock.at[shock_idx].set(shock_size)

    # Storage for responses
    states_path = []
    policies_path = []

    # Apply initial shock
    policy = train_state_obj.apply_fn(train_state_obj.params, state)
    state = econ_model.step(state, policy, initial_shock)
    states_path.append(state)
    policies_path.append(train_state_obj.apply_fn(train_state_obj.params, state))

    # Simulate forward with no further shocks
    zero_shock = jnp.zeros(2, dtype=econ_model.precision)
    for _ in range(n_periods - 1):
        policy = train_state_obj.apply_fn(train_state_obj.params, state)
        state = econ_model.step(state, policy, zero_shock)
        states_path.append(state)
        policies_path.append(train_state_obj.apply_fn(train_state_obj.params, state))

    # Stack into arrays
    states_path = jnp.stack(states_path)
    policies_path = jnp.stack(policies_path)

    # Get aggregates
    aggregates = econ_model.get_aggregates(policies_path, states_path)

    return aggregates


def plot_impulse_responses(ir_productivity, ir_monetary, n_periods, save_dir=None):
    """
    Plot impulse responses to productivity and monetary policy shocks.

    Args:
        ir_productivity: IRFs to productivity shock
        ir_monetary: IRFs to monetary policy shock
        n_periods: Number of periods
        save_dir: Directory to save plots (optional)
    """
    periods = jnp.arange(n_periods)

    # Variables to plot
    variables = [
        ("y_gap", "Output Gap (ỹ)", "percent"),
        ("pi", "Inflation (π)", "percent"),
        ("i", "Nominal Interest Rate (i)", "percent"),
        ("r", "Real Interest Rate (r)", "percent"),
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Impulse Responses - Three-Equation NK Model (Galí Ch. 3)", fontsize=14, fontweight="bold")

    # Color scheme
    colors = {"deqn": "#2E86AB", "zero": "#888888"}

    # Plot productivity shock responses (top row)
    for idx, (var_name, var_label, unit) in enumerate(variables):
        ax = axes[0, idx]
        values = ir_productivity[var_name] * 100  # Convert to percent

        ax.plot(periods, values, color=colors["deqn"], linewidth=2, label="DEQN")
        ax.axhline(y=0, color=colors["zero"], linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Quarters")
        ax.set_ylabel(f"{var_label} (%)")
        ax.set_title(f"{var_label}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_periods - 1)

        if idx == 0:
            ax.set_ylabel("Productivity Shock\n" + f"{var_label} (%)")

    # Plot monetary policy shock responses (bottom row)
    for idx, (var_name, var_label, unit) in enumerate(variables):
        ax = axes[1, idx]
        values = ir_monetary[var_name] * 100  # Convert to percent

        ax.plot(periods, values, color=colors["deqn"], linewidth=2, label="DEQN")
        ax.axhline(y=0, color=colors["zero"], linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Quarters")
        ax.set_ylabel(f"{var_label} (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_periods - 1)

        if idx == 0:
            ax.set_ylabel("Monetary Policy Shock\n" + f"{var_label} (%)")

    # Add row labels
    axes[0, 0].annotate(
        "Productivity\nShock (a)",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )
    axes[1, 0].annotate(
        "Monetary Policy\nShock (v)",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save as PNG
        save_path_png = os.path.join(save_dir, "impulse_responses.png")
        plt.savefig(save_path_png, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path_png}")
        # Save as PDF
        save_path_pdf = os.path.join(save_dir, "impulse_responses.pdf")
        plt.savefig(save_path_pdf, bbox_inches="tight")
        print(f"  Saved: {save_path_pdf}")

    plt.close(fig)

    return fig


def plot_shock_paths(ir_productivity, ir_monetary, n_periods, econ_model, save_dir=None):
    """
    Plot the shock paths (how a_t and v_t evolve after the initial shock).
    """
    periods = jnp.arange(n_periods)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Shock Dynamics (AR(1) Processes)", fontsize=12, fontweight="bold")

    # Productivity shock path
    ax = axes[0]
    ax.plot(periods, ir_productivity["a"] * 100, color="#2E86AB", linewidth=2)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Quarters")
    ax.set_ylabel("Productivity Shock a_t (%)")
    ax.set_title(f"Productivity Shock (ρ_a = {float(econ_model.rho_a):.2f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_periods - 1)

    # Monetary policy shock path
    ax = axes[1]
    ax.plot(periods, ir_monetary["v"] * 100, color="#E94F37", linewidth=2)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Quarters")
    ax.set_ylabel("Monetary Policy Shock v_t (%)")
    ax.set_title(f"Monetary Policy Shock (ρ_v = {float(econ_model.rho_v):.2f})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_periods - 1)

    plt.tight_layout()

    if save_dir:
        # Save as PNG
        save_path_png = os.path.join(save_dir, "shock_paths.png")
        plt.savefig(save_path_png, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path_png}")
        # Save as PDF
        save_path_pdf = os.path.join(save_dir, "shock_paths.pdf")
        plt.savefig(save_path_pdf, bbox_inches="tight")
        print(f"  Saved: {save_path_pdf}")

    plt.close(fig)

    return fig


def print_impact_effects(ir_productivity, ir_monetary):
    """Print the impact (period 0) effects of each shock."""
    print("\n" + "=" * 60)
    print("IMPACT EFFECTS (Period 0)")
    print("=" * 60)

    print("\nProductivity Shock (1 std dev increase in a_t):")
    print(f"  Output Gap (ỹ):      {float(ir_productivity['y_gap'][0]) * 100:>8.4f}%")
    print(f"  Inflation (π):       {float(ir_productivity['pi'][0]) * 100:>8.4f}%")
    print(f"  Nominal Rate (i):    {float(ir_productivity['i'][0]) * 100:>8.4f}%")
    print(f"  Real Rate (r):       {float(ir_productivity['r'][0]) * 100:>8.4f}%")
    print(f"  Natural Rate (r^n):  {float(ir_productivity['r_n'][0]) * 100:>8.4f}%")

    print("\nMonetary Policy Shock (1 std dev increase in v_t):")
    print(f"  Output Gap (ỹ):      {float(ir_monetary['y_gap'][0]) * 100:>8.4f}%")
    print(f"  Inflation (π):       {float(ir_monetary['pi'][0]) * 100:>8.4f}%")
    print(f"  Nominal Rate (i):    {float(ir_monetary['i'][0]) * 100:>8.4f}%")
    print(f"  Real Rate (r):       {float(ir_monetary['r'][0]) * 100:>8.4f}%")
    print(f"  Natural Rate (r^n):  {float(ir_monetary['r_n'][0]) * 100:>8.4f}%")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_model(config, econ_model):
    """Train the NK model and return the trained state."""
    precision = econ_model.precision

    print("\n" + "=" * 60)
    print("Training NK Model")
    print("=" * 60)

    # Create neural network
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
    print("  Compiling...")
    epoch_train_fn = jax.jit(create_epoch_train_fn(econ_model, config))
    eval_fn = jax.jit(create_eval_fn(econ_model, config))

    # Warmup compilation
    t0 = time()
    _ = epoch_train_fn(train_state_obj, rng_epoch)
    _ = eval_fn(train_state_obj, rng_eval)
    print(f"  Compilation time: {time() - t0:.2f}s")

    # Training loop
    print(f"\n{'Epoch':>6} {'Loss':>12} {'Mean Acc':>12} {'Min Acc':>12}")
    print("-" * 48)

    t_start = time()

    for epoch in range(1, config["n_epochs"] + 1):
        train_state_obj, rng_epoch, _ = epoch_train_fn(train_state_obj, rng_epoch)

        if epoch % 10 == 0 or epoch == 1:
            eval_metrics = eval_fn(train_state_obj, rng_eval)
            mean_loss = float(eval_metrics[0])
            mean_acc = float(eval_metrics[1])
            min_acc = float(eval_metrics[2])
            print(f"{epoch:>6} {mean_loss:>12.6f} {mean_acc:>12.4f} {min_acc:>12.4f}")

    print("-" * 48)
    print(f"Training completed in {time() - t_start:.1f}s")
    print(f"Final accuracy: {mean_acc:.4f}")

    return train_state_obj


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 60)
    print("NK Model Analysis: Impulse Responses")
    print("Based on Galí Ch. 3")
    print("=" * 60)

    # Precision setup
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Create output directory
    analysis_dir = os.path.join(base_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Create economic model
    print("\nModel Parameters (Galí notation):")
    econ_model = Model(
        precision=precision,
        double_precision=config["double_precision"],
    )
    print(f"  β (discount):     {float(econ_model.beta):.4f}")
    print(f"  σ (CRRA):         {float(econ_model.sigma):.4f}")
    print(f"  κ (NKPC slope):   {float(econ_model.kappa):.4f}")
    print(f"  φ_π (Taylor π):   {float(econ_model.phi_pi):.4f}")
    print(f"  φ_y (Taylor ỹ):   {float(econ_model.phi_y):.4f}")
    print(f"  ρ_a (persist a):  {float(econ_model.rho_a):.4f}")
    print(f"  ρ_v (persist v):  {float(econ_model.rho_v):.4f}")
    print(f"  σ_a (std a):      {float(econ_model.sigma_a):.4f}")
    print(f"  σ_v (std v):      {float(econ_model.sigma_v):.4f}")

    # Train model
    train_state_obj = train_model(config, econ_model)

    # Compute impulse responses
    print("\n" + "=" * 60)
    print("Computing Impulse Responses")
    print("=" * 60)

    n_periods = config["ir_periods"]
    shock_size = config["shock_size"]

    print(f"  Shock size: {shock_size} std deviations")
    print(f"  Horizon: {n_periods} quarters")

    # IRF to productivity shock (index 0)
    print("\n  Computing productivity shock IRF...")
    ir_productivity = compute_impulse_response(
        train_state_obj, econ_model, shock_idx=0, shock_size=shock_size, n_periods=n_periods
    )

    # IRF to monetary policy shock (index 1)
    print("  Computing monetary policy shock IRF...")
    ir_monetary = compute_impulse_response(
        train_state_obj, econ_model, shock_idx=1, shock_size=shock_size, n_periods=n_periods
    )

    # Print impact effects
    print_impact_effects(ir_productivity, ir_monetary)

    # Plot results
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_impulse_responses(ir_productivity, ir_monetary, n_periods, save_dir=analysis_dir)
    plot_shock_paths(ir_productivity, ir_monetary, n_periods, econ_model, save_dir=analysis_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {analysis_dir}")

    return {
        "train_state": train_state_obj,
        "ir_productivity": ir_productivity,
        "ir_monetary": ir_monetary,
        "econ_model": econ_model,
    }


if __name__ == "__main__":
    results = main()

