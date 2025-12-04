#!/usr/bin/env python3
"""
Run model specification test on RbcProdNet_Dec2025.
"""

import os
import sys

# Setup path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Enable float64 BEFORE importing JAX
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import scipy.io as sio

from DEQN.econ_models.RbcProdNet_Dec2025.model import Model
from DEQN.tests.test_model_specification import test_model_specification


def main():
    print("Loading model data...")
    model_dir = os.path.join(repo_root, "DEQN/econ_models/RbcProdNet_Dec2025")
    model_path = os.path.join(model_dir, "model_data.mat")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = sio.loadmat(model_path, simplify_cells=True)
    print("Model data loaded successfully.")

    # Setup precision
    precision = jnp.float64
    n_sectors = model_data["SolData"]["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    state_ss = jnp.concatenate([model_data["SolData"]["k_ss"], a_ss])

    # Create economic model
    print("Creating economic model...")
    econ_model = Model(
        parameters=model_data["SolData"]["parameters"],
        state_ss=state_ss,
        policies_ss=model_data["SolData"]["policies_ss"],
        state_sd=model_data["SolData"]["states_sd"],
        policies_sd=model_data["SolData"]["policies_sd"],
        double_precision=True,
    )
    print(f"Model created: {n_sectors} sectors, {econ_model.dim_states} states, {econ_model.dim_policies} policies")

    # Run specification test
    print("\nRunning model specification test...")
    results = test_model_specification(econ_model, rtol=1e-6, atol=1e-8, verbose=True)

    return results


if __name__ == "__main__":
    main()

