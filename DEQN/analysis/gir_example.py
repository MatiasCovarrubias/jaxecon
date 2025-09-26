#!/usr/bin/env python3
"""
Example usage of the Generalized Impulse Response (GIR) analysis.

This file demonstrates how to integrate GIR analysis into the RBC ProdNet analysis workflow.
It shows the configuration and usage pattern for the create_GIR_fn function.
"""


# Example of how to integrate GIR into the analysis script
def create_gir_config():
    """Create configuration for GIR analysis."""
    return {
        # Number of points to sample from ergodic distribution
        "n_draws": 100,
        # Length of trajectories to simulate
        "trajectory_length": 50,
        # Size of TFP shock (0.2 = 20% decrease)
        "tfp_shock_size": 0.2,
        # Which sectors to shock (None = all sectors)
        # You can specify specific sectors like [0, 5, 10] to only shock those
        "sectors_to_shock": None,  # or [0, 5, 10, 15, 20] for specific sectors
        # Random seed for reproducibility
        "seed": 42,
    }


def example_gir_integration():
    """
    Example of how to integrate GIR analysis into the main analysis workflow.

    This would be added to the main analysis script after the stochastic steady state calculation.
    """

    # This is how you would modify the Analysis_Sep23_2025.py script:

    print("Example GIR Integration:")
    print("=" * 40)

    # 1. Import the GIR function (add this to imports)
    print("1. Add import: from DEQN.analysis.GIR import create_GIR_fn")

    # 2. Create GIR configuration (add this to create_analysis_config())
    print("2. Add GIR config to analysis_config:")
    gir_config = create_gir_config()
    print(f"   gir_config = {gir_config}")

    # 3. Create GIR function (add this after stoch_ss_fn creation)
    print("3. Create GIR function:")
    print("   gir_fn = jax.jit(create_GIR_fn(econ_model, gir_config))")

    # 4. Calculate GIR (add this in the data collection loop)
    print("4. Calculate GIR for each experiment:")
    print("   gir_results = gir_fn(simul_obs, train_state)")

    # 5. Process and save results
    print("5. Process GIR results:")
    print("   # gir_results is a dictionary with keys for each shocked sector")
    print("   # Each entry contains 'gir_observations' and 'gir_policies' arrays")
    print("   # Shape: gir_observations[T+1, n_states], gir_policies[T, n_policies]")

    print("\nExample configuration for specific sectors:")
    specific_config = create_gir_config()
    specific_config["sectors_to_shock"] = [0, 10, 20, 30]  # Only shock 4 sectors
    specific_config["n_draws"] = 50  # Fewer draws for faster computation
    print(f"   {specific_config}")


def example_usage_code():
    """
    Example code snippet that would be added to the main analysis script.
    """

    code_snippet = """
# Add this import at the top of Analysis_Sep23_2025.py
from DEQN.analysis.GIR import create_GIR_fn

# Add this to create_analysis_config() function
def create_analysis_config():
    return {
        # ... existing config ...
        
        # GIR configuration
        "gir_n_draws": 100,
        "gir_trajectory_length": 50,
        "gir_tfp_shock_size": 0.2,
        "gir_sectors_to_shock": None,  # None for all sectors, or [0, 5, 10] for specific
        "gir_seed": 42,
    }

# Add this in the main() function after creating stoch_ss_fn
def main():
    # ... existing code ...
    
    # Create GIR function
    gir_config = {
        "n_draws": analysis_config["gir_n_draws"],
        "trajectory_length": analysis_config["gir_trajectory_length"],
        "tfp_shock_size": analysis_config["gir_tfp_shock_size"],
        "sectors_to_shock": analysis_config["gir_sectors_to_shock"],
        "seed": analysis_config["gir_seed"],
    }
    gir_fn = jax.jit(create_GIR_fn(econ_model, gir_config))
    
    # Storage for GIR data
    gir_data = {}
    
    # Add this in the data collection loop for each experiment
    for experiment_label, exp_data in experiments_data.items():
        # ... existing simulation and stochastic SS code ...
        
        # 4. Calculate and store GIR
        print(f"    Calculating GIR...")
        gir_results = gir_fn(simul_obs, train_state)
        gir_data[experiment_label] = gir_results
        
        # Print summary of GIR results
        n_sectors_shocked = len(gir_results)
        print(f"    GIR computed for {n_sectors_shocked} sectors")
        
        # Example: Print maximum impulse response magnitude for first sector
        if gir_results:
            first_sector = list(gir_results.keys())[0]
            max_response = jnp.max(jnp.abs(gir_results[first_sector]["gir_observations"]))
            print(f"    Max response magnitude for {first_sector}: {max_response:.6f}")
    
    # Save GIR results (add this after other table generation)
    print("Saving GIR results...")
    gir_save_path = os.path.join(tables_dir, "gir_results.pkl")
    with open(gir_save_path, "wb") as f:
        pickle.dump(gir_data, f)
    print(f"GIR results saved to: {gir_save_path}")
"""

    return code_snippet


if __name__ == "__main__":
    example_gir_integration()
    print("\n" + "=" * 60)
    print("CODE SNIPPET TO ADD TO ANALYSIS SCRIPT:")
    print("=" * 60)
    print(example_usage_code())
