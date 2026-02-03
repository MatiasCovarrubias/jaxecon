"""
Aggregation correction utilities for comparing log-linear and nonlinear simulations.

The issue: In the nonlinear model, we aggregate variables using ergodic (mean) prices
from the simulation. This changes the "steady state" reference for aggregates:
- MATLAB/Dynare: Yagg_ss = Y_ss @ P_ss (steady-state prices)
- Nonlinear: Yagg_ss_ergodic = Y_ss @ P_ergodic (ergodic prices)

To make comparisons consistent, we need to recenter all simulations to use
the same steady state definition (the ergodic-price aggregated one).

Correction formula:
    x_logdev_corrected = x_logdev + log(SS_old) - log(SS_new)

This applies to price-weighted aggregates: Yagg, Kagg, Iagg, Magg.
CES aggregates (Cagg, Lagg) don't need correction as they're quantity-only.

For log-linear solutions, all variables are lognormally distributed:
    log(X/X_ss) ~ N(0, σ²)
    X ~ LogNormal(log(X_ss), σ)
This allows generating synthetic samples from theoretical statistics.
"""

import jax.numpy as jnp
from jax import random


def compute_ergodic_steady_state(
    policies_ss: jnp.ndarray,
    state_ss: jnp.ndarray,
    P_ergodic: jnp.ndarray,
    Pk_ergodic: jnp.ndarray,
    Pm_ergodic: jnp.ndarray,
    n_sectors: int,
) -> dict:
    """
    Compute "proper" steady state aggregates using ergodic price weights.

    The MATLAB steady state computes aggregates as:
        Yagg_ss = P_ss @ Y_ss
        Iagg_ss = Pk_ss @ I_ss
        Magg_ss = Pm_ss @ M_ss
        Kagg_ss = Pk_ss @ K_ss

    This function recomputes them using ergodic prices:
        Yagg_ss_ergodic = P_ergodic @ Y_ss
        etc.

    Args:
        policies_ss: Steady state policies in log (from MATLAB)
        state_ss: Steady state states in log (from MATLAB)
        P_ergodic: Mean output prices in levels from ergodic distribution
        Pk_ergodic: Mean capital prices in levels from ergodic distribution
        Pm_ergodic: Mean intermediate prices in levels from ergodic distribution
        n_sectors: Number of sectors

    Returns:
        Dictionary with both old (MATLAB) and new (ergodic) steady state aggregates
    """
    n = n_sectors

    # Extract steady state quantities in levels
    policies_ss_levels = jnp.exp(policies_ss)

    # Sectoral variables at SS
    Y_ss = policies_ss_levels[10 * n : 11 * n]
    I_ss = policies_ss_levels[6 * n : 7 * n]
    M_ss = policies_ss_levels[4 * n : 5 * n]
    K_ss = jnp.exp(state_ss[:n])

    # Old SS prices (from MATLAB)
    P_ss = policies_ss_levels[8 * n : 9 * n]
    Pk_ss = policies_ss_levels[2 * n : 3 * n]
    Pm_ss = policies_ss_levels[3 * n : 4 * n]

    # Old steady state aggregates (MATLAB definition)
    Yagg_ss_old = Y_ss @ P_ss
    Iagg_ss_old = I_ss @ Pk_ss
    Magg_ss_old = M_ss @ Pm_ss
    Kagg_ss_old = K_ss @ Pk_ss

    # New steady state aggregates (ergodic price definition)
    Yagg_ss_new = Y_ss @ P_ergodic
    Iagg_ss_new = I_ss @ Pk_ergodic
    Magg_ss_new = M_ss @ Pm_ergodic
    Kagg_ss_new = K_ss @ Pk_ergodic

    # CES aggregates don't change (they're quantity-only)
    Cagg_ss = policies_ss_levels[11 * n]
    Lagg_ss = policies_ss_levels[11 * n + 1]

    return {
        # Old definitions (MATLAB)
        "Yagg_ss_old": Yagg_ss_old,
        "Iagg_ss_old": Iagg_ss_old,
        "Magg_ss_old": Magg_ss_old,
        "Kagg_ss_old": Kagg_ss_old,
        # New definitions (ergodic prices)
        "Yagg_ss_new": Yagg_ss_new,
        "Iagg_ss_new": Iagg_ss_new,
        "Magg_ss_new": Magg_ss_new,
        "Kagg_ss_new": Kagg_ss_new,
        # CES aggregates (unchanged)
        "Cagg_ss": Cagg_ss,
        "Lagg_ss": Lagg_ss,
        # Correction factors (log differences)
        "Yagg_correction": jnp.log(Yagg_ss_old) - jnp.log(Yagg_ss_new),
        "Iagg_correction": jnp.log(Iagg_ss_old) - jnp.log(Iagg_ss_new),
        "Magg_correction": jnp.log(Magg_ss_old) - jnp.log(Magg_ss_new),
        "Kagg_correction": jnp.log(Kagg_ss_old) - jnp.log(Kagg_ss_new),
    }


def recenter_analysis_variables(
    analysis_vars: dict,
    ss_corrections: dict,
) -> dict:
    """
    Recenter analysis variables to use ergodic-price aggregated steady state.

    The correction is: x_logdev_corrected = x_logdev + log(SS_old) - log(SS_new)

    This ensures that when SS_old prices are used in the simulation but we want
    the reference to be SS_new (ergodic prices), the log deviations are properly shifted.

    Args:
        analysis_vars: Dictionary of analysis variables (log deviations from SS)
        ss_corrections: Dictionary from compute_ergodic_steady_state

    Returns:
        Dictionary with corrected analysis variables
    """
    corrected = {}

    # Map analysis variable names to their corrections
    correction_map = {
        "Agg. Output": "Yagg_correction",
        "Agg. Investment": "Iagg_correction",
        "Agg. Intermediates": "Magg_correction",
        "Agg. Capital": "Kagg_correction",
    }

    for var_name, values in analysis_vars.items():
        if var_name in correction_map:
            correction = ss_corrections[correction_map[var_name]]
            corrected[var_name] = values + correction
        else:
            # No correction needed for CES aggregates (Cagg, Lagg) and Utility
            corrected[var_name] = values

    return corrected


def compute_ergodic_prices_from_simulation(
    simul_policies: jnp.ndarray,
    policies_ss: jnp.ndarray,
    n_sectors: int,
) -> tuple:
    """
    Compute ergodic price levels from simulation policies.

    Args:
        simul_policies: Simulation policies in log deviations (T, n_policy_vars)
        policies_ss: Steady state policies in log
        n_sectors: Number of sectors

    Returns:
        Tuple of (P_ergodic, Pk_ergodic, Pm_ergodic) in levels
    """
    n = n_sectors

    # Get mean log deviations
    simul_policies_mean = jnp.mean(simul_policies, axis=0)

    # SS prices in levels
    P_ss = jnp.exp(policies_ss[8 * n : 9 * n])
    Pk_ss = jnp.exp(policies_ss[2 * n : 3 * n])
    Pm_ss = jnp.exp(policies_ss[3 * n : 4 * n])

    # Mean prices in levels: P_level = P_ss * exp(mean_logdev)
    P_ergodic = P_ss * jnp.exp(simul_policies_mean[8 * n : 9 * n])
    Pk_ergodic = Pk_ss * jnp.exp(simul_policies_mean[2 * n : 3 * n])
    Pm_ergodic = Pm_ss * jnp.exp(simul_policies_mean[3 * n : 4 * n])

    return P_ergodic, Pk_ergodic, Pm_ergodic


def process_simulation_with_consistent_aggregation(
    simul_data: jnp.ndarray,
    policies_ss: jnp.ndarray,
    state_ss: jnp.ndarray,
    P_ergodic: jnp.ndarray,
    Pk_ergodic: jnp.ndarray,
    Pm_ergodic: jnp.ndarray,
    n_sectors: int,
    burn_in: int = 0,
    source_label: str = "simulation",
) -> dict:
    """
    Process simulation data with consistent ergodic-price aggregation.

    This function:
    1. Extracts sectoral variables from simulation
    2. Aggregates using ergodic prices
    3. Computes SS aggregates using the same ergodic prices
    4. Returns log deviations that are consistently defined

    Args:
        simul_data: Simulation output (n_vars, T) in log levels
        policies_ss: Steady state policies in log
        state_ss: Steady state states in log
        P_ergodic: Ergodic output prices in levels
        Pk_ergodic: Ergodic capital prices in levels
        Pm_ergodic: Ergodic intermediate prices in levels
        n_sectors: Number of sectors
        burn_in: Number of initial periods to discard
        source_label: Label for debugging output

    Returns:
        Dictionary with analysis variables (log deviations from ergodic-price SS)
    """
    n = n_sectors
    idx = _get_variable_indices(n)

    # Apply burn-in
    n_periods = simul_data.shape[1]
    if burn_in >= n_periods:
        burn_in = n_periods // 10
    simul = simul_data[:, burn_in:]

    # Extract sectoral variables in levels
    K = jnp.exp(simul[idx["k"][0] : idx["k"][1], :])  # (n_sectors, n_periods)
    Y = jnp.exp(simul[idx["y"][0] : idx["y"][1], :])
    Inv = jnp.exp(simul[idx["i"][0] : idx["i"][1], :])
    M = jnp.exp(simul[idx["m"][0] : idx["m"][1], :])

    # CES aggregates from simulation (these are consistent definitions)
    Cagg = jnp.exp(simul[idx["cagg"], :])
    Lagg = jnp.exp(simul[idx["lagg"], :])

    # Aggregate using ergodic prices
    Kagg = K.T @ Pk_ergodic
    Yagg = Y.T @ P_ergodic
    Iagg = Inv.T @ Pk_ergodic
    Magg = M.T @ Pm_ergodic

    # Compute SS aggregates using SAME ergodic prices
    policies_ss_levels = jnp.exp(policies_ss)
    K_ss = jnp.exp(state_ss[:n])
    Y_ss = policies_ss_levels[10 * n : 11 * n]
    I_ss = policies_ss_levels[6 * n : 7 * n]
    M_ss = policies_ss_levels[4 * n : 5 * n]
    Cagg_ss = policies_ss_levels[11 * n]
    Lagg_ss = policies_ss_levels[11 * n + 1]

    Kagg_ss = K_ss @ Pk_ergodic
    Yagg_ss = Y_ss @ P_ergodic
    Iagg_ss = I_ss @ Pk_ergodic
    Magg_ss = M_ss @ Pm_ergodic

    # Compute log deviations
    result = {
        "Agg. Consumption": jnp.log(Cagg) - jnp.log(Cagg_ss),
        "Agg. Labor": jnp.log(Lagg) - jnp.log(Lagg_ss),
        "Agg. Capital": jnp.log(Kagg) - jnp.log(Kagg_ss),
        "Agg. Output": jnp.log(Yagg) - jnp.log(Yagg_ss),
        "Agg. Intermediates": jnp.log(Magg) - jnp.log(Magg_ss),
        "Agg. Investment": jnp.log(Iagg) - jnp.log(Iagg_ss),
    }

    return result


def _get_variable_indices(n_sectors: int) -> dict:
    """Get variable indices for Dynare simulation data (0-indexed for Python)."""
    n = n_sectors
    return {
        "k": (0, n),
        "a": (n, 2 * n),
        "c": (2 * n, 3 * n),
        "l": (3 * n, 4 * n),
        "pk": (4 * n, 5 * n),
        "pm": (5 * n, 6 * n),
        "m": (6 * n, 7 * n),
        "mout": (7 * n, 8 * n),
        "i": (8 * n, 9 * n),
        "iout": (9 * n, 10 * n),
        "p": (10 * n, 11 * n),
        "q": (11 * n, 12 * n),
        "y": (12 * n, 13 * n),
        "cagg": 13 * n,
        "lagg": 13 * n + 1,
        "yagg": 13 * n + 2,
        "iagg": 13 * n + 3,
        "magg": 13 * n + 4,
    }


def generate_loglinear_samples_from_theostats(
    theo_stats: dict,
    n_samples: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Generate synthetic log-linear samples from theoretical statistics.

    For log-linear solutions, all aggregate variables in log-deviations are
    normally distributed around 0 with variance from TheoStats:
        log(X/X_ss) ~ N(0, σ²)

    This function generates samples from these distributions for comparison
    with nonlinear simulation results.

    Args:
        theo_stats: Dictionary with theoretical statistics from MATLAB/Dynare.
                   Expected keys: sigma_C_agg, sigma_L_agg, sigma_VA_agg,
                                  sigma_I_agg, sigma_M_agg
                   Optional: rho_VA_agg (autocorrelation), var_cov_agg (correlations)
        n_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with analysis variables (log deviations from SS):
            - "Agg. Consumption": samples of log(C/C_ss)
            - "Agg. Labor": samples of log(L/L_ss)
            - "Agg. Output": samples of log(Y/Y_ss)
            - "Agg. Investment": samples of log(I/I_ss)
            - "Agg. Intermediates": samples of log(M/M_ss)
    """
    key = random.PRNGKey(seed)

    # Map theo_stats keys to analysis variable names
    # Log-linear: log(X/X_ss) ~ N(0, sigma²)
    var_mapping = {
        "Agg. Consumption": "sigma_C_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. Output": "sigma_VA_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Intermediates": "sigma_M_agg",
    }

    result = {}

    # Check if we have correlation structure
    if "var_cov_agg" in theo_stats and theo_stats["var_cov_agg"] is not None:
        # Generate correlated samples using the full covariance matrix
        # Order in var_cov: cagg, lagg, yagg, iagg, magg
        var_cov = jnp.array(theo_stats["var_cov_agg"])

        # Generate multivariate normal samples
        # MVN samples: z = L @ epsilon, where L L^T = Sigma
        L = jnp.linalg.cholesky(var_cov)
        key, subkey = random.split(key)
        epsilon = random.normal(subkey, shape=(n_samples, 5))
        samples = epsilon @ L.T

        # Assign to result dict
        result["Agg. Consumption"] = samples[:, 0]
        result["Agg. Labor"] = samples[:, 1]
        result["Agg. Output"] = samples[:, 2]
        result["Agg. Investment"] = samples[:, 3]
        result["Agg. Intermediates"] = samples[:, 4]
    else:
        # Generate independent samples (no correlation structure)
        for var_name, sigma_key in var_mapping.items():
            if sigma_key in theo_stats:
                sigma = theo_stats[sigma_key]
                key, subkey = random.split(key)
                # Log-linear: log(X/X_ss) ~ N(0, sigma²)
                samples = random.normal(subkey, shape=(n_samples,)) * sigma
                result[var_name] = samples
            else:
                print(f"  Warning: {sigma_key} not found in theo_stats, skipping {var_name}")

    return result


def get_loglinear_distribution_params(theo_stats: dict) -> dict:
    """
    Get distribution parameters for analytical log-linear density plots.

    For log-linear solutions:
        log(X/X_ss) ~ N(0, σ²)  [log deviation is normal with mean 0]

    This returns (mean=0, std=sigma) for each aggregate variable,
    which can be used to plot analytical PDFs.

    Args:
        theo_stats: Dictionary with theoretical statistics from MATLAB/Dynare

    Returns:
        Dictionary mapping variable names to (mean, std) tuples
    """
    var_mapping = {
        "Agg. Consumption": "sigma_C_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. Output": "sigma_VA_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Intermediates": "sigma_M_agg",
    }

    result = {}
    for var_name, sigma_key in var_mapping.items():
        if sigma_key in theo_stats:
            sigma = theo_stats[sigma_key]
            # Log-linear: mean of log deviation is 0, std is sigma
            result[var_name] = {"mean": 0.0, "std": float(sigma)}

    return result


def create_theoretical_descriptive_stats(theo_stats: dict, label: str = "Log-Linear (Theoretical)") -> dict:
    """
    Create pre-computed descriptive statistics from theoretical moments.

    For log-linear (first-order) approximation, all variables are lognormally distributed:
        log(X/X_ss) ~ N(0, σ²)

    This means:
        - Mean of log deviation = 0 (by construction)
        - Std = σ (from TheoStats)
        - Skewness = 0 (normal distribution)
        - Excess Kurtosis = 0 (normal distribution)

    Args:
        theo_stats: Dictionary with theoretical statistics from MATLAB/Dynare
                   Expected keys: sigma_C_agg, sigma_L_agg, sigma_VA_agg, sigma_I_agg, sigma_M_agg
        label: Label for this experiment in the output

    Returns:
        Dictionary in format {label: {var_label: {"Mean": val, "Sd": val, "Skewness": val, "Excess Kurtosis": val}}}
        Ready to pass to create_descriptive_stats_table as theoretical_stats
    """
    var_mapping = {
        "Agg. Consumption": "sigma_C_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. Output": "sigma_VA_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Intermediates": "sigma_M_agg",
    }

    result = {}
    for var_name, sigma_key in var_mapping.items():
        if sigma_key in theo_stats:
            sigma = float(theo_stats[sigma_key])
            result[var_name] = {
                "Mean": 0.0,  # Log-linear mean is 0 by construction
                "Sd": sigma * 100,  # Convert to percentage
                "Skewness": 0.0,  # Normal distribution
                "Excess Kurtosis": 0.0,  # Normal distribution (excess kurtosis = 0)
            }

    return {label: result}


def create_perfect_foresight_descriptive_stats(
    determ_stats: dict,
    label: str = "Perfect Foresight",
    n_sectors: int = 37,
) -> dict:
    """
    Create pre-computed descriptive statistics from perfect foresight (deterministic) statistics.

    Supports two formats:
    1. Legacy format (pre-Feb 2026): Individual fields like Cagg_volatility, Cagg_mean_logdev
    2. New format (Feb 2026+): Full vectors policies_mean, policies_std

    Note: Skewness and Kurtosis are not available for perfect foresight (would need simulation).

    Args:
        determ_stats: Dictionary with deterministic statistics from MATLAB.
                     Legacy format: Cagg_volatility, Lagg_volatility, Cagg_mean_logdev, etc.
                     New format: policies_mean (412x1), policies_std (412x1)
        label: Label for this experiment in the output
        n_sectors: Number of sectors (default 37)

    Returns:
        Dictionary in format {label: {var_label: {"Mean": val, "Sd": val, "Skewness": val, "Excess Kurtosis": val}}}
        Ready to pass to create_descriptive_stats_table as theoretical_stats
    """
    result = {}
    n = n_sectors

    # Check if we have new format (policies_mean/policies_std vectors)
    has_new_format = "policies_std" in determ_stats or "policies_mean" in determ_stats

    if has_new_format:
        # New format: extract from full policy vectors
        # Policy indexing (for 37 sectors, 412 = 11*37 + 5 policies):
        #   policies[11*n+0] = cagg
        #   policies[11*n+1] = lagg
        #   policies[11*n+2] = yagg
        #   policies[11*n+3] = iagg
        #   policies[11*n+4] = magg
        agg_indices = {
            "Agg. Consumption": 11 * n,
            "Agg. Labor": 11 * n + 1,
            "Agg. Output": 11 * n + 2,
            "Agg. Investment": 11 * n + 3,
            "Agg. Intermediates": 11 * n + 4,
        }

        policies_mean = determ_stats.get("policies_mean")
        policies_std = determ_stats.get("policies_std")

        for var_name, idx in agg_indices.items():
            stats_dict = {}

            if policies_mean is not None and idx < len(policies_mean):
                stats_dict["Mean"] = float(policies_mean[idx]) * 100  # Convert to %

            if policies_std is not None and idx < len(policies_std):
                stats_dict["Sd"] = float(policies_std[idx]) * 100  # Convert to %
            else:
                stats_dict["Sd"] = float("nan")

            # Skewness and kurtosis not available in new format
            stats_dict["Skewness"] = float("nan")
            stats_dict["Excess Kurtosis"] = float("nan")

            if stats_dict:
                result[var_name] = stats_dict
    else:
        # Legacy format: use individual field names
        var_mapping = {
            "Agg. Consumption": ("Cagg_mean_logdev", "Cagg_volatility", "Cagg_skewness", "Cagg_kurtosis"),
            "Agg. Labor": ("Lagg_mean_logdev", "Lagg_volatility", "Lagg_skewness", "Lagg_kurtosis"),
            "Agg. Output": ("Yagg_mean_logdev", "Yagg_volatility", "Yagg_skewness", "Yagg_kurtosis"),
            "Agg. Investment": ("Iagg_mean_logdev", "Iagg_volatility", "Iagg_skewness", "Iagg_kurtosis"),
            "Agg. Intermediates": ("Magg_mean_logdev", "Magg_volatility", "Magg_skewness", "Magg_kurtosis"),
        }

        for var_name, (mean_key, vol_key, skew_key, kurt_key) in var_mapping.items():
            stats_dict = {}

            # Get mean (stored as fraction, convert to %)
            if mean_key in determ_stats:
                stats_dict["Mean"] = float(determ_stats[mean_key]) * 100

            # Get volatility/std (stored as fraction, convert to %)
            if vol_key and vol_key in determ_stats:
                stats_dict["Sd"] = float(determ_stats[vol_key]) * 100
            else:
                stats_dict["Sd"] = float("nan")

            # Get skewness (if available from MATLAB)
            if skew_key and skew_key in determ_stats:
                stats_dict["Skewness"] = float(determ_stats[skew_key])
            else:
                stats_dict["Skewness"] = float("nan")

            # Get excess kurtosis (if available from MATLAB)
            if kurt_key and kurt_key in determ_stats:
                stats_dict["Excess Kurtosis"] = float(determ_stats[kurt_key])
            else:
                stats_dict["Excess Kurtosis"] = float("nan")

            if stats_dict:
                result[var_name] = stats_dict

    return {label: result}
