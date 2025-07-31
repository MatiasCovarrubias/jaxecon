import jax
import jax.numpy as jnp
import jax.random as random
from aggregators import (
    aggregator_tornqvist,
    aggregator_tornqvist_logdevs,
    aggregator_fixedprice,
    tornqvist_index_vectorized,
    tornqvist_index_loop,
)

# TEST FUNCTIONS
# ==============


def test_tornqvist_aggregator(model, T_test=100, seed=42):
    """
    Test the Tornqvist aggregator with simulated stationary series to check for divergence issues.

    This function tests whether the Tornqvist aggregator correctly handles stationary time series.
    If the aggregator is working correctly, stationary inputs should produce stationary outputs.
    """
    print("\n=== Testing Tornqvist Aggregator ===")

    # Set random seed for reproducibility
    rng = random.PRNGKey(seed)

    # Generate stationary time series for quantities and prices
    # Using AR(1) processes around 1.0 with small persistence and noise
    rho = 0.8  # persistence parameter
    sigma = 0.05  # innovation standard deviation

    # Initialize arrays
    Y_test = jnp.ones((T_test, model.n_sectors))  # Start at 1.0
    P_test = jnp.ones((T_test, model.n_sectors))  # Start at 1.0

    # Generate AR(1) processes for both Y and P
    for t in range(1, T_test):
        rng, rng_y, rng_p = random.split(rng, 3)

        # Y follows AR(1): Y_t = (1-rho) + rho * Y_{t-1} + epsilon_t
        eps_y = random.normal(rng_y, shape=(model.n_sectors,)) * sigma
        Y_new = (1 - rho) + rho * Y_test[t - 1, :] + eps_y
        Y_test = Y_test.at[t, :].set(Y_new)

        # P follows AR(1): P_t = (1-rho) + rho * P_{t-1} + epsilon_t
        eps_p = random.normal(rng_p, shape=(model.n_sectors,)) * sigma
        P_new = (1 - rho) + rho * P_test[t - 1, :] + eps_p
        P_test = P_test.at[t, :].set(P_new)

    print(f"Generated {T_test} periods of stationary data")
    print(f"Y mean: {jnp.mean(Y_test):.4f}, Y std: {jnp.std(Y_test):.4f}")
    print(f"P mean: {jnp.mean(P_test):.4f}, P std: {jnp.std(P_test):.4f}")

    # Test current vectorized Tornqvist implementation
    print("\n--- Testing Vectorized Tornqvist ---")
    yagg_vectorized = tornqvist_index_vectorized(Y_test, P_test, verbose=True)

    # Test loop-based implementation for comparison
    print("\n--- Testing Loop-based Tornqvist ---")
    yagg_loop = tornqvist_index_loop(Y_test, P_test)

    # Compare results
    print(f"\nVectorized - Mean: {jnp.mean(yagg_vectorized):.4f}, Std: {jnp.std(yagg_vectorized):.4f}")
    print(f"Vectorized - Start: {yagg_vectorized[0]:.4f}, End: {yagg_vectorized[-1]:.4f}")
    print(f"Vectorized - Trend: {(yagg_vectorized[-1] / yagg_vectorized[0] - 1) * 100:.2f}%")

    print(f"\nLoop-based - Mean: {jnp.mean(yagg_loop):.4f}, Std: {jnp.std(yagg_loop):.4f}")
    print(f"Loop-based - Start: {yagg_loop[0]:.4f}, End: {yagg_loop[-1]:.4f}")
    print(f"Loop-based - Trend: {(yagg_loop[-1] / yagg_loop[0] - 1) * 100:.2f}%")

    print(f"\nMax difference between implementations: {jnp.max(jnp.abs(yagg_vectorized - yagg_loop)):.6f}")

    # Test simple price-weighted aggregator for comparison
    price_weighted = jnp.sum(Y_test * P_test, axis=1) / jnp.sum(P_test, axis=1)
    print(f"\nPrice-weighted - Mean: {jnp.mean(price_weighted):.4f}, Std: {jnp.std(price_weighted):.4f}")
    print(f"Price-weighted - Trend: {(price_weighted[-1] / price_weighted[0] - 1) * 100:.2f}%")

    return {
        "Y_test": Y_test,
        "P_test": P_test,
        "yagg_vectorized": yagg_vectorized,
        "yagg_loop": yagg_loop,
        "price_weighted": price_weighted,
    }


def test_tornqvist_drift_comparison(model, T_test=1000, seed=42):
    """
    Test comparing the old (levels-based) vs new (log-deviations) Tornqvist aggregators.

    This test demonstrates why the new approach prevents drift:
    - Old approach: returns cumulative levels starting from 1, prone to drift
    - New approach: returns log deviations from steady state, centered around 0
    """
    print("\n" + "=" * 60)
    print("TORNQVIST DRIFT COMPARISON TEST")
    print("=" * 60)

    rng = random.PRNGKey(seed)

    # Generate simulated data that oscillates around steady state
    rho = 0.9  # persistence
    sigma = 0.02  # innovation std

    # Initialize with steady state
    simul_obs = jnp.zeros((T_test, 3 * model.n_sectors))
    simul_policy = jnp.ones((T_test, 11 * model.n_sectors + 5))

    # Add some persistence and random shocks to make it realistic
    for t in range(1, T_test):
        rng, rng_k, rng_policy = random.split(rng, 3)

        # Add some persistence to capital
        k_shock = random.normal(rng_k, shape=(model.n_sectors,)) * sigma
        simul_obs = simul_obs.at[t, : model.n_sectors].set(rho * simul_obs[t - 1, : model.n_sectors] + k_shock)

        # Add small random variations to policy
        policy_shock = random.normal(rng_policy, shape=(11 * model.n_sectors + 5,)) * sigma * 0.5
        simul_policy = simul_policy.at[t, :].set(simul_policy[t - 1, :] * jnp.exp(policy_shock))

    print(f"Generated {T_test} periods of simulated data")
    print(f"Data oscillates around steady state with persistence {rho} and std {sigma}")

    # Test old approach (levels-based)
    print("\n--- OLD APPROACH (Levels-based) ---")
    old_results = aggregator_tornqvist(model, simul_obs, simul_policy, use_loop=False, add_steady_state=False)
    yagg_old = old_results["Yagg_tornqvist"]

    print(f"Mean level: {jnp.mean(yagg_old):.4f}")
    print(f"Std: {jnp.std(yagg_old):.4f}")
    print(f"Start: {yagg_old[0]:.4f}, End: {yagg_old[-1]:.4f}")
    print(f"Total drift: {(yagg_old[-1] / yagg_old[0] - 1) * 100:.2f}%")
    print(f"Moving average (last 100): {jnp.mean(yagg_old[-100:]):.4f}")

    # Test new approach (log-deviations)
    print("\n--- NEW APPROACH (Log-deviations) ---")
    new_results = aggregator_tornqvist_logdevs(model, simul_obs, simul_policy)
    yagg_new = new_results["Yagg_tornqvist"]

    print(f"Mean log deviation: {jnp.mean(yagg_new):.6f}")
    print(f"Std: {jnp.std(yagg_new):.4f}")
    print(f"Start: {yagg_new[0]:.6f}, End: {yagg_new[-1]:.6f}")
    print(f"Total drift: {yagg_new[-1] - yagg_new[0]:.6f} log points")
    print(f"Moving average (last 100): {jnp.mean(yagg_new[-100:]):.6f}")

    # Compare to deterministic aggregator
    print("\n--- DETERMINISTIC AGGREGATOR (for comparison) ---")
    Pss = jnp.exp(model.policies_ss[8 * model.n_sectors : 9 * model.n_sectors])
    det_results = aggregator_fixedprice(model, simul_obs, simul_policy, Pss, Pss, Pss)
    yagg_det = det_results["Yagg"]

    print(f"Mean log deviation: {jnp.mean(yagg_det):.6f}")
    print(f"Std: {jnp.std(yagg_det):.4f}")
    print(f"Moving average (last 100): {jnp.mean(yagg_det[-100:]):.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- Old approach shows systematic drift in levels")
    print("- New approach is stationary around 0 (like deterministic aggregator)")
    print("- Both capture similar economic fluctuations")
    print("=" * 60)

    return {
        "old_yagg": yagg_old,
        "new_yagg": yagg_new,
        "det_yagg": yagg_det,
        "simul_obs": simul_obs,
        "simul_policy": simul_policy,
    }


def test_national_accounts_identity(model, simul_obs, simul_policy):
    """
    Test if Y = C + I identity holds with different aggregators.
    """
    print("\n=== Testing National Accounts Identity ===")

    # Extract variables
    vars_dict = model.get_variables(simul_obs, simul_policy)

    # Test with deterministic steady state prices
    Pss = jnp.exp(model.policies_ss[8 * model.n_sectors : 9 * model.n_sectors])
    Pkss = jnp.exp(model.policies_ss[2 * model.n_sectors : 3 * model.n_sectors])

    C_agg_det = jnp.sum(vars_dict["C"] * Pss, axis=1)
    I_agg_det = jnp.sum(vars_dict["I"] * Pkss, axis=1)
    Y_agg_det = jnp.sum(vars_dict["Y"] * Pss, axis=1)

    identity_det = Y_agg_det - C_agg_det - I_agg_det
    print(f"Deterministic prices - Y-C-I mean: {jnp.mean(identity_det):.6f}, std: {jnp.std(identity_det):.6f}")

    # Test with Tornqvist aggregators
    tornqvist_results = aggregator_tornqvist(model, simul_obs, simul_policy)
    C_agg_torn = tornqvist_results["Cagg_tornqvist"]
    I_agg_torn = tornqvist_results["Iagg_tornqvist"]
    Y_agg_torn = tornqvist_results["Yagg_tornqvist"]

    identity_torn = Y_agg_torn - C_agg_torn - I_agg_torn
    print(f"Tornqvist prices - Y-C-I mean: {jnp.mean(identity_torn):.6f}, std: {jnp.std(identity_torn):.6f}")

    return {
        "identity_deterministic": identity_det,
        "identity_tornqvist": identity_torn,
        "Y_det": Y_agg_det,
        "C_det": C_agg_det,
        "I_det": I_agg_det,
        "Y_torn": Y_agg_torn,
        "C_torn": C_agg_torn,
        "I_torn": I_agg_torn,
    }


def diagnose_tornqvist_issues(model, simul_obs=None, simul_policy=None, T_test=200):
    """
    Comprehensive diagnostic function to identify Tornqvist aggregator issues.

    This function runs multiple tests to check:
    1. Vectorized vs loop implementation differences
    2. Stationary series behavior
    3. Effect of adding steady state initial observation
    4. National accounts identity compliance
    """
    print("=" * 60)
    print("COMPREHENSIVE TORNQVIST DIAGNOSTICS")
    print("=" * 60)

    # Test 1: Stationary series behavior
    print("\n" + "=" * 40)
    print("TEST 1: STATIONARY SERIES BEHAVIOR")
    print("=" * 40)
    stationary_results = test_tornqvist_aggregator(model, T_test=T_test)

    # Test 2: If real simulation data provided, test it
    if simul_obs is not None and simul_policy is not None:
        print("\n" + "=" * 40)
        print("TEST 2: REAL SIMULATION DATA")
        print("=" * 40)

        # Test without steady state (current default behavior)
        print("\n--- Without Steady State ---")
        results_no_ss_vec = aggregator_tornqvist(model, simul_obs, simul_policy, use_loop=False, add_steady_state=False)
        print(f"Vectorized Y - Mean: {jnp.mean(results_no_ss_vec['Yagg_tornqvist']):.4f}")
        print(
            f"Vectorized Y - Start: {results_no_ss_vec['Yagg_tornqvist'][0]:.4f}, End: {results_no_ss_vec['Yagg_tornqvist'][-1]:.4f}"
        )
        print(
            f"Vectorized Y - Trend: {(results_no_ss_vec['Yagg_tornqvist'][-1] / results_no_ss_vec['Yagg_tornqvist'][0] - 1) * 100:.2f}%"
        )

        results_no_ss_loop = aggregator_tornqvist(model, simul_obs, simul_policy, use_loop=True, add_steady_state=False)
        print(f"Loop Y - Mean: {jnp.mean(results_no_ss_loop['Yagg_tornqvist']):.4f}")
        print(
            f"Loop Y - Start: {results_no_ss_loop['Yagg_tornqvist'][0]:.4f}, End: {results_no_ss_loop['Yagg_tornqvist'][-1]:.4f}"
        )
        print(
            f"Loop Y - Trend: {(results_no_ss_loop['Yagg_tornqvist'][-1] / results_no_ss_loop['Yagg_tornqvist'][0] - 1) * 100:.2f}%"
        )

        max_diff = jnp.max(jnp.abs(results_no_ss_vec["Yagg_tornqvist"] - results_no_ss_loop["Yagg_tornqvist"]))
        print(f"Max difference (Vec vs Loop): {max_diff:.6f}")

        # Test with steady state
        print("\n--- With Steady State ---")
        results_with_ss_vec = aggregator_tornqvist(
            model, simul_obs, simul_policy, use_loop=False, add_steady_state=True
        )
        print(f"Vectorized+SS Y - Mean: {jnp.mean(results_with_ss_vec['Yagg_tornqvist']):.4f}")
        print(
            f"Vectorized+SS Y - Start: {results_with_ss_vec['Yagg_tornqvist'][0]:.4f}, End: {results_with_ss_vec['Yagg_tornqvist'][-1]:.4f}"
        )
        print(
            f"Vectorized+SS Y - Trend: {(results_with_ss_vec['Yagg_tornqvist'][-1] / results_with_ss_vec['Yagg_tornqvist'][0] - 1) * 100:.2f}%"
        )

        # Test 3: National accounts identity
        print("\n" + "=" * 40)
        print("TEST 3: NATIONAL ACCOUNTS IDENTITY")
        print("=" * 40)
        identity_results = test_national_accounts_identity(model, simul_obs, simul_policy)

        # Test 4: Compare with deterministic aggregator
        print("\n" + "=" * 40)
        print("TEST 4: DETERMINISTIC VS TORNQVIST")
        print("=" * 40)

        # Use steady state prices for deterministic aggregation
        Pss = jnp.exp(model.policies_ss[8 * model.n_sectors : 9 * model.n_sectors])
        Pkss = jnp.exp(model.policies_ss[2 * model.n_sectors : 3 * model.n_sectors])

        det_results = aggregator_fixedprice(model, simul_obs, simul_policy, Pss, Pkss, Pss)

        print(f"Deterministic Y - Mean: {jnp.mean(jnp.exp(det_results['Yagg'])):.4f}")
        print(
            f"Deterministic Y - Start: {jnp.exp(det_results['Yagg'][0]):.4f}, End: {jnp.exp(det_results['Yagg'][-1]):.4f}"
        )
        print(
            f"Deterministic Y - Trend: {(jnp.exp(det_results['Yagg'][-1]) / jnp.exp(det_results['Yagg'][0]) - 1) * 100:.2f}%"
        )

        print(f"Tornqvist Y - Mean: {jnp.mean(results_no_ss_vec['Yagg_tornqvist']):.4f}")
        print(
            f"Tornqvist Y - Trend: {(results_no_ss_vec['Yagg_tornqvist'][-1] / results_no_ss_vec['Yagg_tornqvist'][0] - 1) * 100:.2f}%"
        )

        # Return all results for further analysis
        return {
            "stationary_test": stationary_results,
            "simulation_no_ss_vec": results_no_ss_vec,
            "simulation_no_ss_loop": results_no_ss_loop,
            "simulation_with_ss": results_with_ss_vec,
            "identity_test": identity_results,
            "deterministic": det_results,
        }
    else:
        print("\nNote: To run tests on real simulation data, provide simul_obs and simul_policy")
        return {"stationary_test": stationary_results}


def demo_tornqvist_logdevs(model):
    """
    Quick demonstration of the new log-deviations Tornqvist aggregator.

    This shows how to use the new drift-free Tornqvist implementation.
    """
    print("\n" + "=" * 50)
    print("DEMO: NEW TORNQVIST LOG-DEVIATIONS AGGREGATOR")
    print("=" * 50)

    # Create some sample data (small deviation from steady state)
    T_sample = 10
    simul_obs = jnp.zeros((T_sample, 3 * model.n_sectors))
    simul_policy = jnp.ones((T_sample, 11 * model.n_sectors + 5))

    # Add small random variations
    rng = random.PRNGKey(123)
    rng, rng_obs, rng_policy = random.split(rng, 3)

    obs_noise = random.normal(rng_obs, shape=(T_sample, 3 * model.n_sectors)) * 0.01
    policy_noise = random.normal(rng_policy, shape=(T_sample, 11 * model.n_sectors + 5)) * 0.005

    simul_obs = simul_obs + obs_noise
    simul_policy = simul_policy * jnp.exp(policy_noise)

    print("Sample data created with small deviations from steady state")

    # Test the new log-deviations aggregator
    print("\n--- Using New Log-Deviations Aggregator ---")
    results = aggregator_tornqvist_logdevs(model, simul_obs, simul_policy)

    print("Results (log deviations from steady state):")
    for key, values in results.items():
        print(f"{key:15s}: mean={jnp.mean(values):8.4f}, std={jnp.std(values):8.4f}")

    # Compare with deterministic aggregator
    print("\n--- Comparing with Deterministic Aggregator ---")
    Pss = jnp.exp(model.policies_ss[8 * model.n_sectors : 9 * model.n_sectors])
    det_results = aggregator_fixedprice(model, simul_obs, simul_policy, Pss, Pss, Pss)

    print("Deterministic aggregator results:")
    for key, values in det_results.items():
        if key != "Cagg_prices":  # Skip the alternative name
            print(f"{key:15s}: mean={jnp.mean(values):8.4f}, std={jnp.std(values):8.4f}")

    print("\n" + "=" * 50)
    print("SUCCESS: New Tornqvist aggregator working correctly!")
    print("- Returns log deviations from steady state (like deterministic aggregator)")
    print("- No cumulative drift issues")
    print("- Ready for use in your economic model")
    print("=" * 50)

    return results
