#!/usr/bin/env python3
"""
Value Function Iteration with JAX - Educational Script

This script demonstrates JAX fundamentals through Value Function Iteration:
- jax.jit: Just-in-time compilation
- jax.vmap: Automatic vectorization
- jax.pmap: Multi-device parallelization (TPU/multi-GPU)
- jax.lax.scan: Efficient loops

The economic problem is the classic income fluctuation model:
    max E[Σ β^t u(c_t)]  subject to  c_t + a_{t+1} ≤ R*a_t + y_t

Usage:
    LOCAL:
        python VFI/vfi.py

    COLAB:
        Copy this script into a Colab cell and run.
        The script auto-detects the environment.

Based on: https://notes.quantecon.org/submission/622ed4daf57192000f918c61
"""

import os
import sys
import time

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
    print("Installing dependencies...")
    import subprocess

    subprocess.run(["pip", "install", "-q", "quantecon"], check=True)

# ============================================================================
# IMPORTS
# ============================================================================

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from scipy import stats

# Optional: quantecon for Tauchen method (fallback implemented below)
try:
    import quantecon as qe

    HAS_QUANTECON = True
except ImportError:
    HAS_QUANTECON = False
    print("  Note: quantecon not installed, using built-in Tauchen method")

# ============================================================================
# CONFIGURATION
# ============================================================================

config = {
    # Grid parameters (adjust scale for larger experiments)
    "scale": 1,  # Multiplier for grid sizes (1=small, 32=large)
    # Model parameters
    "R": 1.1,  # Gross interest rate
    "beta": 0.99,  # Discount factor
    "gamma": 2.5,  # Risk aversion (CRRA utility)
    # Income process (AR(1) in logs)
    "rho": 0.9,  # Persistence
    "sigma": 0.1,  # Volatility
    # Asset grid bounds
    "a_min": 0.01,
    "a_max": 2.0,
    # VFI parameters
    "tol": 1e-6,
    "max_iter": 100,
    # Benchmark settings
    "run_benchmarks": False,  # Set True to run benchmarks
    "benchmark_scales": [1, 2],
    "benchmark_iterations": 5,
}


# ============================================================================
# TAUCHEN METHOD (fallback if quantecon not installed)
# ============================================================================


def tauchen(n: int, rho: float, sigma: float, m: float = 3.0):
    """
    Tauchen's method to discretize AR(1) process: y' = rho * y + sigma * e

    Args:
        n: Number of grid points
        rho: Persistence parameter
        sigma: Standard deviation of innovations
        m: Number of standard deviations for grid bounds

    Returns:
        state_values: Grid points for y
        P: Transition matrix P[i,j] = Prob(y'=y_j | y=y_i)
    """
    # Unconditional std of y
    sigma_y = sigma / np.sqrt(1 - rho**2)

    # Grid
    y_max = m * sigma_y
    y_min = -y_max
    state_values = np.linspace(y_min, y_max, n)
    step = state_values[1] - state_values[0]

    # Transition matrix
    P = np.zeros((n, n))
    for i in range(n):
        # Expected value of y' given y_i
        mu = rho * state_values[i]

        # Probabilities
        P[i, 0] = stats.norm.cdf((state_values[0] + step / 2 - mu) / sigma)
        P[i, n - 1] = 1 - stats.norm.cdf((state_values[n - 1] - step / 2 - mu) / sigma)

        for j in range(1, n - 1):
            z_low = (state_values[j] - step / 2 - mu) / sigma
            z_high = (state_values[j] + step / 2 - mu) / sigma
            P[i, j] = stats.norm.cdf(z_high) - stats.norm.cdf(z_low)

    return state_values, P


# ============================================================================
# DEVICE DETECTION
# ============================================================================


def detect_devices():
    """Detect available JAX devices and print info."""
    devices = jax.devices()
    n_devices = len(devices)
    device_type = devices[0].platform.upper()

    print(f"\n{'='*60}")
    print(f"  JAX Device Info")
    print(f"{'='*60}")
    print(f"  Backend: {device_type}")
    print(f"  Device count: {n_devices}")
    print(f"  Devices: {[str(d) for d in devices]}")
    print(f"  JAX version: {jax.__version__}")
    print(f"{'='*60}\n")

    return devices, n_devices, device_type


# ============================================================================
# MODEL SETUP
# ============================================================================


def create_model(config: dict, precision=jnp.float32):
    """Create the economic model with grids and parameters."""
    scale = config["scale"]

    # Grid sizes
    a_size = 1024 * scale
    y_size = 128 * scale

    # Asset grid
    a_grid = jnp.linspace(config["a_min"], config["a_max"], a_size, dtype=precision)

    # Income grid via Tauchen method
    if HAS_QUANTECON:
        mc = qe.tauchen(n=y_size, rho=config["rho"], sigma=config["sigma"])
        y_vals, P_vals = mc.state_values, mc.P
    else:
        y_vals, P_vals = tauchen(n=y_size, rho=config["rho"], sigma=config["sigma"])

    y_grid = jnp.array(np.exp(y_vals), dtype=precision)
    P = jnp.array(P_vals, dtype=precision)

    # Model dictionary
    model = {
        "params": {
            "R": config["R"],
            "beta": config["beta"],
            "gamma": config["gamma"],
        },
        "grids": {
            "a": a_grid,
            "y": y_grid,
            "ap": a_grid,  # Same grid for next-period assets
        },
        "P": P,  # Transition matrix
        "sizes": {
            "a": a_size,
            "y": y_size,
        },
        "indices": {
            "a": jnp.arange(a_size),
            "y": jnp.arange(y_size),
            "ap": jnp.arange(a_size),
        },
        # Pre-shaped grids for manual vectorization
        "batched_grids": {
            "a": a_grid.reshape(a_size, 1, 1),
            "y": y_grid.reshape(1, y_size, 1),
            "ap": a_grid.reshape(1, 1, a_size),
            "P": P.reshape(y_size, y_size, 1),
        },
    }

    return model


# ============================================================================
# BELLMAN OPERATORS - DIFFERENT VECTORIZATION STRATEGIES
# ============================================================================


def get_T_naive(model: dict):
    """
    Naive implementation using Python loops.
    This is SLOW but shows the basic algorithm clearly.
    """
    params = model["params"]
    a_grid = model["grids"]["a"]
    y_grid = model["grids"]["y"]
    ap_grid = model["grids"]["ap"]
    P = model["P"]
    R, beta, gamma = params["R"], params["beta"], params["gamma"]

    def u(c):
        return c ** (1 - gamma) / (1 - gamma)

    def T_naive(v):
        a_size, y_size = v.shape
        v_new = np.empty_like(v)

        for i, a in enumerate(a_grid):
            for j, y in enumerate(y_grid):
                v_max = -np.inf
                for k, ap in enumerate(ap_grid):
                    c = R * a + y - ap
                    if c > 0:
                        val = u(c) + beta * np.dot(v[k, :], P[j, :])
                        v_max = max(v_max, val)
                v_new[i, j] = v_max
        return v_new

    return T_naive


def get_T_manual_vectorization(model: dict):
    """
    Manual vectorization using array broadcasting.

    Key insight: Reshape arrays to broadcast correctly:
    - a:  (a_size, 1, 1)
    - y:  (1, y_size, 1)
    - ap: (1, 1, ap_size)
    - P:  (y_size, y_size, 1)

    This creates a 3D array of all state-action combinations.
    """
    params = model["params"]
    bg = model["batched_grids"]
    R, beta, gamma = params["R"], params["beta"], params["gamma"]

    a = bg["a"]
    y = bg["y"]
    ap = bg["ap"]
    P = bg["P"]

    def u(c):
        return c ** (1 - gamma) / (1 - gamma)

    def T_manual(v):
        # v @ P gives expected continuation value: (a_size, y_size, 1)
        Ev = jnp.dot(v, P)

        # Broadcasting: c has shape (a_size, y_size, ap_size)
        c = R * a + y - ap

        # Value for each (a, y, ap) combination
        values = jnp.where(c > 0, u(c) + beta * Ev, -jnp.inf)

        # Maximize over ap (last axis)
        return jnp.max(values, axis=2)

    return T_manual


def get_T_vmap(model: dict):
    """
    Automatic vectorization using jax.vmap.

    vmap transforms a function that operates on single elements
    into one that operates on batches. We apply it progressively:
    1. vmap over ap (actions)
    2. vmap over y (income states)
    3. vmap over a (asset states)
    """
    params = model["params"]
    grids = model["grids"]
    P = model["P"]
    indices = model["indices"]
    R, beta, gamma = params["R"], params["beta"], params["gamma"]

    def u(c):
        return c ** (1 - gamma) / (1 - gamma)

    def T_vmap(v):
        # Value for a single (a, y, ap) combination
        def action_value(a_idx, y_idx, ap_idx):
            c = R * grids["a"][a_idx] + grids["y"][y_idx] - grids["ap"][ap_idx]
            Ev = jnp.dot(v[ap_idx, :], P[y_idx, :])
            return jnp.where(c > 0, u(c) + beta * Ev, -jnp.inf)

        # Vectorize over ap (innermost)
        action_values_over_ap = jax.vmap(action_value, in_axes=(None, None, 0))

        # Max over actions for a single state
        def state_value(a_idx, y_idx):
            return jnp.max(action_values_over_ap(a_idx, y_idx, indices["ap"]))

        # Vectorize over y, then over a
        all_state_values = jax.vmap(jax.vmap(state_value, in_axes=(None, 0)), in_axes=(0, None))

        return all_state_values(indices["a"], indices["y"])

    return T_vmap


def get_T_pmap(model: dict, n_devices: int):
    """
    Multi-device parallelization using jax.pmap.

    pmap distributes computation across multiple devices (TPU cores, GPUs).
    We partition the asset grid across devices.

    For TPU v2-8: 8 cores → 8 partitions
    For multi-GPU: n_gpu partitions
    """
    params = model["params"]
    grids = model["grids"]
    P = model["P"]
    indices = model["indices"]
    R, beta, gamma = params["R"], params["beta"], params["gamma"]
    a_size = model["sizes"]["a"]

    # Partition asset indices across devices
    a_partitions = indices["a"].reshape(n_devices, a_size // n_devices)

    def u(c):
        return c ** (1 - gamma) / (1 - gamma)

    def T_partition(a_partition, v):
        """Update value function for a partition of asset states."""

        def action_value(a_idx, y_idx, ap_idx):
            c = R * grids["a"][a_idx] + grids["y"][y_idx] - grids["ap"][ap_idx]
            Ev = jnp.dot(v[ap_idx, :], P[y_idx, :])
            return jnp.where(c > 0, u(c) + beta * Ev, -jnp.inf)

        action_values_over_ap = jax.vmap(action_value, in_axes=(None, None, 0))

        def state_value(a_idx, y_idx):
            return jnp.max(action_values_over_ap(a_idx, y_idx, indices["ap"]))

        all_state_values = jax.vmap(jax.vmap(state_value, in_axes=(None, 0)), in_axes=(0, None))

        return all_state_values(a_partition, indices["y"])

    # pmap across devices, v is broadcast (same on all devices)
    T_pmapped = jax.pmap(T_partition, in_axes=(0, None))

    def T_pmap_wrapper(v):
        # Run on all devices, then reshape back
        result = T_pmapped(a_partitions, v)
        return result.reshape(a_size, model["sizes"]["y"])

    return T_pmap_wrapper, a_partitions


# ============================================================================
# VFI WITH JAX.LAX.SCAN
# ============================================================================


def get_vfi_scan(model: dict):
    """
    Full VFI using jax.lax.scan for efficient iteration.

    jax.lax.scan is like a for loop but:
    1. Compiles the entire loop into XLA
    2. Avoids Python overhead per iteration
    3. Can be differentiated through
    """
    params = model["params"]
    grids = model["grids"]
    P = model["P"]
    indices = model["indices"]
    R, beta, gamma = params["R"], params["beta"], params["gamma"]

    def u(c):
        return c ** (1 - gamma) / (1 - gamma)

    def bellman_step(v, _):
        """Single Bellman update, compatible with scan."""

        def action_value(a_idx, y_idx, ap_idx):
            c = R * grids["a"][a_idx] + grids["y"][y_idx] - grids["ap"][ap_idx]
            Ev = jnp.dot(v[ap_idx, :], P[y_idx, :])
            return jnp.where(c > 0, u(c) + beta * Ev, -jnp.inf)

        action_values_over_ap = jax.vmap(action_value, in_axes=(None, None, 0))

        def state_value(a_idx, y_idx):
            return jnp.max(action_values_over_ap(a_idx, y_idx, indices["ap"]))

        all_state_values = jax.vmap(jax.vmap(state_value, in_axes=(None, 0)), in_axes=(0, None))

        new_v = all_state_values(indices["a"], indices["y"])
        error = jnp.max(jnp.abs(new_v - v))

        return new_v, error

    def vfi_scan(v_init, n_iterations):
        """Run VFI for fixed number of iterations using scan."""
        final_v, errors = jax.lax.scan(bellman_step, v_init, None, length=n_iterations)
        return final_v, errors

    return vfi_scan


# ============================================================================
# TIMING UTILITIES
# ============================================================================


def time_function(fn, *args, n_runs=10, warmup=2):
    """Time a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 60)
    print("  VALUE FUNCTION ITERATION WITH JAX")
    print("  Educational demonstration of JAX fundamentals")
    print("=" * 60)

    # Detect devices
    devices, n_devices, device_type = detect_devices()

    # Create model
    print(f"Creating model with scale={config['scale']}...")
    model = create_model(config)
    a_size, y_size = model["sizes"]["a"], model["sizes"]["y"]
    state_space_size = a_size * y_size
    print(f"  Asset grid: {a_size} points")
    print(f"  Income grid: {y_size} points")
    print(f"  State space: {state_space_size:,} points")

    # Initial value function
    v_init = jnp.zeros((a_size, y_size))

    # =========================================================================
    # DEMO 1: Manual Vectorization
    # =========================================================================
    print("\n" + "-" * 60)
    print("  1. MANUAL VECTORIZATION (array broadcasting)")
    print("-" * 60)

    T_manual = get_T_manual_vectorization(model)
    T_manual_jit = jax.jit(T_manual)

    # Compile
    print("  Compiling...")
    _ = T_manual_jit(v_init).block_until_ready()

    # Time
    mean_time, std_time = time_function(T_manual_jit, v_init)
    print(f"  Time per update: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")

    # =========================================================================
    # DEMO 2: Automatic Vectorization with vmap
    # =========================================================================
    print("\n" + "-" * 60)
    print("  2. AUTOMATIC VECTORIZATION (jax.vmap)")
    print("-" * 60)

    T_vmap = get_T_vmap(model)
    T_vmap_jit = jax.jit(T_vmap)

    # Compile
    print("  Compiling...")
    _ = T_vmap_jit(v_init).block_until_ready()

    # Time
    mean_time, std_time = time_function(T_vmap_jit, v_init)
    print(f"  Time per update: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")

    # =========================================================================
    # DEMO 3: Multi-device Parallelization with pmap
    # =========================================================================
    if n_devices > 1:
        print("\n" + "-" * 60)
        print(f"  3. MULTI-DEVICE PARALLELIZATION (jax.pmap, {n_devices} devices)")
        print("-" * 60)

        T_pmap, a_partitions = get_T_pmap(model, n_devices)

        # For pmap, we need to compile differently
        T_partition_base = lambda a_part, v: get_T_pmap(model, n_devices)[0](v)

        # Compile
        print("  Compiling...")
        _ = T_pmap(v_init).block_until_ready()

        # Time
        mean_time, std_time = time_function(T_pmap, v_init)
        print(f"  Time per update: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    else:
        print("\n  [Skipping pmap demo - only 1 device available]")

    # =========================================================================
    # DEMO 4: Full VFI with jax.lax.scan
    # =========================================================================
    print("\n" + "-" * 60)
    print("  4. FULL VFI WITH JAX.LAX.SCAN")
    print("-" * 60)

    vfi_scan = get_vfi_scan(model)
    vfi_scan_jit = jax.jit(vfi_scan, static_argnums=(1,))

    n_iterations = config["max_iter"]
    print(f"  Running {n_iterations} iterations...")

    # Compile and run
    start = time.perf_counter()
    v_final, errors = vfi_scan_jit(v_init, n_iterations)
    v_final.block_until_ready()
    total_time = time.perf_counter() - start

    # Check convergence
    final_error = float(errors[-1])
    converged = final_error < config["tol"]

    print(f"  Total time: {total_time:.2f} s")
    print(f"  Time per iteration: {total_time/n_iterations*1000:.2f} ms")
    print(f"  Final error: {final_error:.2e}")
    print(f"  Converged: {'Yes ✓' if converged else 'No (increase max_iter)'}")

    # =========================================================================
    # BENCHMARKS
    # =========================================================================
    if config["run_benchmarks"]:
        print("\n" + "=" * 60)
        print("  BENCHMARKS")
        print("=" * 60)

        results = {
            "Scale": [],
            "State Space": [],
            "Manual (ms)": [],
            "vmap (ms)": [],
        }
        if n_devices > 1:
            results["pmap (ms)"] = []

        for scale in config["benchmark_scales"]:
            bench_config = config.copy()
            bench_config["scale"] = scale
            bench_model = create_model(bench_config)

            a_s, y_s = bench_model["sizes"]["a"], bench_model["sizes"]["y"]
            v_bench = jnp.zeros((a_s, y_s))

            results["Scale"].append(scale)
            results["State Space"].append(f"{a_s * y_s:,}")

            # Manual
            T_m = jax.jit(get_T_manual_vectorization(bench_model))
            _ = T_m(v_bench).block_until_ready()
            t, _ = time_function(T_m, v_bench, n_runs=config["benchmark_iterations"])
            results["Manual (ms)"].append(f"{t*1000:.2f}")

            # vmap
            T_v = jax.jit(get_T_vmap(bench_model))
            _ = T_v(v_bench).block_until_ready()
            t, _ = time_function(T_v, v_bench, n_runs=config["benchmark_iterations"])
            results["vmap (ms)"].append(f"{t*1000:.2f}")

            # pmap
            if n_devices > 1 and a_s % n_devices == 0:
                T_p, _ = get_T_pmap(bench_model, n_devices)
                _ = T_p(v_bench).block_until_ready()
                t, _ = time_function(T_p, v_bench, n_runs=config["benchmark_iterations"])
                results["pmap (ms)"].append(f"{t*1000:.2f}")

            print(f"  Scale {scale}: done")

        # Print table
        print("\n  Results:")
        print("-" * 60)
        header = " | ".join(f"{k:>12}" for k in results.keys())
        print(f"  {header}")
        print("  " + "-" * len(header))
        for i in range(len(results["Scale"])):
            row = " | ".join(f"{results[k][i]:>12}" for k in results.keys())
            print(f"  {row}")

    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)

    return v_final, errors


if __name__ == "__main__":
    main()

