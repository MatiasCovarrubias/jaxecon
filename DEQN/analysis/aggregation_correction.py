"""
Aggregation utilities for comparing log-linear and nonlinear simulations.

Current aggregate definitions used for moments/IRs:
- Consumption expenditure: C_exp(t) = sum_j P_j(t) * C_j(t)
- Investment expenditure: I_exp(t) = sum_j P_j(t) * I_j^{out}(t)
- GDP expenditure: GDP_exp(t) = sum_j P_j(t) * [Q_j(t) - M_j^{out}(t)]

All are returned as log deviations from deterministic steady state:
    log(X_t) - log(X_ss)

Utility aggregate consumption Cagg is still available as a separate object
for welfare purposes, but it is not used as the main aggregate consumption
series for moments/IR comparisons.
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
    Compute steady-state auxiliary aggregates/corrections.

    Expenditure aggregates (C_exp, I_exp, GDP_exp) are now computed using
    current prices and deterministic steady-state references directly in
    model/analysis pipelines, so no recentering correction is needed for them.

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

    # Sectoral variables at SS (used for diagnostics/corrections that remain)
    Y_ss = policies_ss_levels[10 * n : 11 * n]
    I_ss = policies_ss_levels[6 * n : 7 * n]
    M_ss = policies_ss_levels[4 * n : 5 * n]
    K_ss = jnp.exp(state_ss[:n])

    # Old SS prices (from MATLAB)
    P_ss = policies_ss_levels[8 * n : 9 * n]
    Pk_ss = policies_ss_levels[2 * n : 3 * n]
    Pm_ss = policies_ss_levels[3 * n : 4 * n]

    # Old steady state aggregates (legacy weighted definitions)
    Yagg_ss_old = Y_ss @ P_ss
    Iagg_ss_old = I_ss @ Pk_ss
    Magg_ss_old = M_ss @ Pm_ss
    Kagg_ss_old = K_ss @ Pk_ss

    # New steady state aggregates (ergodic-price weighted diagnostics)
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

    # Only keep recentering for legacy/diagnostic aggregates.
    # Main expenditure aggregates (Agg. Consumption / Agg. Investment / Agg. Output)
    # should not be shifted here.
    correction_map = {
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
    Process Dynare simulation using expenditure-based aggregate definitions.

    Simulation data is stored as log deviations from steady state:
        simul[i, t] = log(X_i(t)) - log(X_i_ss).

    Main aggregates:
    - C_exp(t)   = sum_j P_j(t) * C_j(t)
    - I_exp(t)   = sum_j P_j(t) * I_j^{out}(t)
    - GDP_exp(t) = sum_j P_j(t) * [Q_j(t) - M_j^{out}(t)]
    and their log deviations from deterministic SS.

    Utility aggregate consumption Cagg and CES labor aggregate Lagg are kept as
    separate diagnostic series.

    Args:
        simul_data: (n_vars, T) log deviations from SS (ModelData_simulation full_simul)
        policies_ss: Steady state policies in log
        state_ss: Steady state states in log (first n_sectors = capital)
        P_ergodic: Output price weights in levels (same as nonlinear model)
        Pk_ergodic: Capital price weights in levels
        Pm_ergodic: Intermediate price weights in levels
        n_sectors: Number of sectors
        burn_in: Number of initial periods to discard
        source_label: Label for debugging output

    Returns:
        Dictionary with analysis variables (log deviations from SS)
    """
    n = n_sectors
    idx = _get_variable_indices(n)

    expected_n_vars = 13 * n + 5
    if simul_data.shape[0] == expected_n_vars:
        pass
    elif simul_data.shape[1] == expected_n_vars:
        simul_data = simul_data.T
    else:
        raise ValueError(
            f"{source_label}: unexpected simulation shape {simul_data.shape}. "
            f"Expected one axis to equal n_vars={expected_n_vars}."
        )

    # MATLAB/Dynare full_simul can be saved either as:
    # - log deviations from SS, or
    # - log levels.
    # Auto-detect and convert log levels -> log deviations when needed.
    ss_full = jnp.concatenate([state_ss, policies_ss])
    dist_to_zero = jnp.mean(jnp.abs(simul_data[:, 0]))
    dist_to_ss = jnp.mean(jnp.abs(simul_data[:, 0] - ss_full))
    if dist_to_ss < dist_to_zero:
        simul_data = simul_data - ss_full[:, None]

    n_periods = simul_data.shape[1]
    if burn_in >= n_periods:
        burn_in = n_periods // 10
    simul = simul_data[:, burn_in:]

    policies_ss_levels = jnp.exp(policies_ss)
    K_ss = jnp.exp(state_ss[:n])
    Y_ss = policies_ss_levels[10 * n : 11 * n]
    I_ss = policies_ss_levels[6 * n : 7 * n]
    M_ss = policies_ss_levels[4 * n : 5 * n]

    log_dev_k = simul[idx["k"][0] : idx["k"][1], :]
    log_dev_c = simul[idx["c"][0] : idx["c"][1], :]
    log_dev_iout = simul[idx["iout"][0] : idx["iout"][1], :]
    log_dev_q = simul[idx["q"][0] : idx["q"][1], :]
    log_dev_mout = simul[idx["mout"][0] : idx["mout"][1], :]
    log_dev_p = simul[idx["p"][0] : idx["p"][1], :]
    log_dev_m = simul[idx["m"][0] : idx["m"][1], :]

    K_levels = K_ss[:, None] * jnp.exp(log_dev_k)
    C_ss = policies_ss_levels[0:n]
    Iout_ss = policies_ss_levels[7 * n : 8 * n]
    Q_ss = policies_ss_levels[9 * n : 10 * n]
    Mout_ss = policies_ss_levels[5 * n : 6 * n]
    P_ss = policies_ss_levels[8 * n : 9 * n]

    C_levels = C_ss[:, None] * jnp.exp(log_dev_c)
    Iout_levels = Iout_ss[:, None] * jnp.exp(log_dev_iout)
    Q_levels = Q_ss[:, None] * jnp.exp(log_dev_q)
    Mout_levels = Mout_ss[:, None] * jnp.exp(log_dev_mout)
    P_levels = P_ss[:, None] * jnp.exp(log_dev_p)

    M_levels = M_ss[:, None] * jnp.exp(log_dev_m)

    Kagg = K_levels.T @ Pk_ergodic
    Cagg_exp = jnp.sum(P_levels * C_levels, axis=0)
    Iagg_exp = jnp.sum(P_levels * Iout_levels, axis=0)
    GDPagg_exp = jnp.sum(P_levels * (Q_levels - Mout_levels), axis=0)
    Magg = M_levels.T @ Pm_ergodic

    Kagg_ss = K_ss @ Pk_ergodic
    Cagg_exp_ss = jnp.sum(P_ss * C_ss)
    Iagg_exp_ss = jnp.sum(P_ss * Iout_ss)
    GDPagg_exp_ss = jnp.sum(P_ss * (Q_ss - Mout_ss))
    Magg_ss = M_ss @ Pm_ergodic

    cagg_logdev = jnp.ravel(simul[idx["cagg"], :])
    lagg_logdev = jnp.ravel(simul[idx["lagg"], :])

    epsilon = jnp.array(1e-12)

    result = {
        "Agg. Consumption": jnp.log(jnp.maximum(Cagg_exp, epsilon)) - jnp.log(jnp.maximum(Cagg_exp_ss, epsilon)),
        "Agg. Consumption (Utility)": cagg_logdev,
        "Agg. Labor": lagg_logdev,
        "Agg. Capital": jnp.log(Kagg) - jnp.log(Kagg_ss),
        "Agg. Output": jnp.log(jnp.maximum(GDPagg_exp, epsilon)) - jnp.log(jnp.maximum(GDPagg_exp_ss, epsilon)),
        "Agg. GDP": jnp.log(jnp.maximum(GDPagg_exp, epsilon)) - jnp.log(jnp.maximum(GDPagg_exp_ss, epsilon)),
        "Agg. Intermediates": jnp.log(Magg) - jnp.log(Magg_ss),
        "Agg. Investment": jnp.log(jnp.maximum(Iagg_exp, epsilon)) - jnp.log(jnp.maximum(Iagg_exp_ss, epsilon)),
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


def create_theoretical_descriptive_stats(theo_stats: dict, label: str = "Log-Linear") -> dict:
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
    model_stats: dict | None = None,
    policies_ss: jnp.ndarray | None = None,
) -> dict:
    """
    Create pre-computed descriptive statistics from perfect foresight (deterministic) statistics.

    Supports:
    1. Legacy format (pre-Feb 2026): Individual fields like Cagg_volatility, Cagg_mean_logdev
    2. New format (Feb 2026+): policies_mean, policies_std, and optionally ModelStats

    When model_stats is provided (Statistics.PerfectForesight.ModelStats), aggregate volatilities
    are taken from it when available (sigma_VA_agg, sigma_L_agg, sigma_I_agg, sigma_M_agg).
    ModelStats does not include sigma_C_agg; Consumption Sd is taken from policies_std.

    policies_mean in MATLAB is the mean over time of log levels; mean log deviation in percent
    is (policies_mean - policies_ss) * 100. When policies_ss is provided, Mean is computed that way.

    Note: Skewness and Kurtosis are not available for perfect foresight (would need simulation).

    Args:
        determ_stats: Dictionary with deterministic statistics from MATLAB.
                     Legacy: Cagg_volatility, Lagg_volatility, etc.
                     New: policies_mean (412x1), policies_std (412x1)
        label: Label for this experiment in the output
        n_sectors: Number of sectors (default 37)
        model_stats: Optional ModelStats struct (Statistics.PerfectForesight.ModelStats).
                     When present, used for sigma_VA_agg, sigma_L_agg, sigma_I_agg, sigma_M_agg.
        policies_ss: Optional steady-state policies (log levels). When provided, Mean for
                     aggregates is (policies_mean - policies_ss) * 100 (mean log deviation in %).

    Returns:
        Dictionary in format {label: {var_label: {"Mean": val, "Sd": val, "Skewness": val, "Excess Kurtosis": val}}}
        Ready to pass to create_descriptive_stats_table as theoretical_stats
    """
    result = {}
    n = n_sectors
    ms = model_stats if model_stats else {}

    model_stats_sd_map = {
        "Agg. Output": "sigma_VA_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Intermediates": "sigma_M_agg",
    }

    has_new_format = "policies_std" in determ_stats or "policies_mean" in determ_stats

    if has_new_format:
        agg_indices = {
            "Agg. Consumption": 11 * n,
            "Agg. Labor": 11 * n + 1,
            "Agg. Output": 11 * n + 2,
            "Agg. Investment": 11 * n + 3,
            "Agg. Intermediates": 11 * n + 4,
        }

        policies_mean = determ_stats.get("policies_mean")
        policies_std = determ_stats.get("policies_std")
        ps = None
        if policies_ss is not None:
            ps = jnp.asarray(policies_ss)
            if hasattr(ps, "shape") and len(ps.shape) > 1:
                ps = jnp.ravel(ps)

        for var_name, idx in agg_indices.items():
            stats_dict = {}

            if policies_mean is not None and idx < len(policies_mean):
                pm = float(policies_mean[idx])
                if ps is not None and idx < len(ps):
                    stats_dict["Mean"] = (pm - float(ps[idx])) * 100
                else:
                    stats_dict["Mean"] = pm * 100

            sd_key = model_stats_sd_map.get(var_name)
            if sd_key and sd_key in ms:
                stats_dict["Sd"] = float(ms[sd_key]) * 100
            elif policies_std is not None and idx < len(policies_std):
                stats_dict["Sd"] = float(policies_std[idx]) * 100
            else:
                stats_dict["Sd"] = float("nan")

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
