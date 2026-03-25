"""
Aggregation utilities for comparing log-linear and nonlinear simulations.

Default behavior reads aggregate policy variables directly from the model output.
Optional behavior (`ergodic_price_aggregation=True`) re-aggregates aggregates
from sectoral objects using fixed ergodic-mean prices.
"""

import jax.numpy as jnp
import numpy as np
from jax import random

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

    # SS prices in levels
    P_ss = jnp.exp(policies_ss[8 * n : 9 * n])
    Pk_ss = jnp.exp(policies_ss[2 * n : 3 * n])
    Pm_ss = jnp.exp(policies_ss[3 * n : 4 * n])

    # Convert each simulated price path to levels first, then take the arithmetic
    # mean so the fixed-price vector matches the intended expenditure weights.
    P_ergodic = jnp.mean(P_ss[None, :] * jnp.exp(simul_policies[:, 8 * n : 9 * n]), axis=0)
    Pk_ergodic = jnp.mean(Pk_ss[None, :] * jnp.exp(simul_policies[:, 2 * n : 3 * n]), axis=0)
    Pm_ergodic = jnp.mean(Pm_ss[None, :] * jnp.exp(simul_policies[:, 3 * n : 4 * n]), axis=0)

    return P_ergodic, Pk_ergodic, Pm_ergodic


def reaggregate_aggregates(
    state_logdev: jnp.ndarray,
    policies_logdev: jnp.ndarray,
    *,
    policies_ss: jnp.ndarray,
    state_ss: jnp.ndarray,
    log_policy_count: int,
    utility_intratemp_idx: int,
    P_weights: jnp.ndarray | None = None,
    Pk_weights: jnp.ndarray | None = None,
) -> dict:
    """Rebuild exposed aggregates from sectoral objects under fixed prices."""
    epsilon = jnp.array(1e-12)
    squeeze_output = policies_logdev.ndim == 1

    state_logdev = jnp.atleast_2d(state_logdev)
    policies_logdev = jnp.atleast_2d(policies_logdev)
    n = state_logdev.shape[1] // 2

    P_ss = jnp.exp(policies_ss[8 * n : 9 * n])
    Pk_ss = jnp.exp(policies_ss[2 * n : 3 * n])

    if P_weights is None:
        P_weights = jnp.zeros_like(P_ss)
    if Pk_weights is None:
        Pk_weights = jnp.zeros_like(Pk_ss)

    P_levels = P_ss * jnp.exp(P_weights)
    Pk_levels = Pk_ss * jnp.exp(Pk_weights)

    policies_notnorm = policies_logdev + policies_ss[None, :]
    policies_levels = jnp.exp(policies_notnorm[:, :log_policy_count])
    policies_ss_levels = jnp.exp(policies_ss[:log_policy_count])

    state_notnorm = state_logdev + state_ss[None, :]
    K = jnp.exp(state_notnorm[:, :n])
    K_ss = jnp.exp(state_ss[:n])

    C = policies_levels[:, :n]
    L = policies_levels[:, n : 2 * n]
    Iout = policies_levels[:, 7 * n : 8 * n]
    Q = policies_levels[:, 9 * n : 10 * n]
    Mout = policies_levels[:, 5 * n : 6 * n]

    C_ss = policies_ss_levels[:n]
    L_ss = policies_ss_levels[n : 2 * n]
    Iout_ss = policies_ss_levels[7 * n : 8 * n]
    Q_ss = policies_ss_levels[9 * n : 10 * n]
    Mout_ss = policies_ss_levels[5 * n : 6 * n]

    result = {
        "Agg. Consumption": jnp.log(jnp.maximum(jnp.sum(P_levels[None, :] * C, axis=1), epsilon))
        - jnp.log(jnp.maximum(jnp.sum(P_levels * C_ss), epsilon)),
        "Agg. Labor": jnp.log(jnp.maximum(jnp.sum(L, axis=1), epsilon)) - jnp.log(jnp.maximum(jnp.sum(L_ss), epsilon)),
        "Agg. Capital": jnp.log(jnp.maximum(jnp.sum(K * Pk_levels[None, :], axis=1), epsilon))
        - jnp.log(jnp.maximum(jnp.sum(K_ss * Pk_levels), epsilon)),
        "Agg. Output": jnp.log(jnp.maximum(jnp.sum(P_levels[None, :] * (Q - Mout), axis=1), epsilon))
        - jnp.log(jnp.maximum(jnp.sum(P_levels * (Q_ss - Mout_ss)), epsilon)),
        "Agg. GDP": jnp.log(jnp.maximum(jnp.sum(P_levels[None, :] * (Q - Mout), axis=1), epsilon))
        - jnp.log(jnp.maximum(jnp.sum(P_levels * (Q_ss - Mout_ss)), epsilon)),
        "Agg. Investment": jnp.log(jnp.maximum(jnp.sum(P_levels[None, :] * Iout, axis=1), epsilon))
        - jnp.log(jnp.maximum(jnp.sum(P_levels * Iout_ss), epsilon)),
        "Intratemporal Utility": policies_logdev[:, utility_intratemp_idx],
    }
    if squeeze_output:
        return {key: jnp.ravel(value)[0] for key, value in result.items()}
    return result


def process_simulation_with_consistent_aggregation(
    simul_data: jnp.ndarray,
    policies_ss: jnp.ndarray,
    state_ss: jnp.ndarray,
    P_ergodic: jnp.ndarray,
    Pk_ergodic: jnp.ndarray,
    n_sectors: int,
    ergodic_price_aggregation: bool = False,
    burn_in: int = 0,
    source_label: str = "simulation",
) -> dict:
    """
    Process Dynare simulation using expenditure-based aggregate definitions.

    Simulation data is stored as log deviations from steady state:
        simul[i, t] = log(X_i(t)) - log(X_i_ss).

    Main aggregates use fixed ergodic-mean prices (matching DEQN aggregation):
    - C_exp(t)   = sum_j P_j^{erg} * C_j(t)
    - I_exp(t)   = sum_j P_j^{erg} * I_j^{out}(t)
    - GDP_exp(t) = sum_j P_j^{erg} * [Q_j(t) - M_j^{out}(t)]
    and their log deviations from the same ergodic-price steady-state reference.

    Default mode reads the aggregate policy tail directly. Optional ergodic mode
    re-aggregates the sectoral objects under fixed ergodic-mean prices.

    Args:
        simul_data: (n_vars, T) log deviations from SS (ModelData_simulation full_simul)
        policies_ss: Steady state policies in log
        state_ss: Steady state states in log (first n_sectors = capital)
        P_ergodic: Output price weights in levels (same as nonlinear model)
        Pk_ergodic: Capital price weights in levels
        n_sectors: Number of sectors
        burn_in: Number of initial periods to discard
        source_label: Label for debugging output

    Returns:
        Dictionary with analysis variables (log deviations from SS)
    """
    n = n_sectors

    expected_n_vars = 13 * n + 8
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

    policy_logdev = simul[2 * n :, :].T
    if not ergodic_price_aggregation:
        return {
            "Agg. Consumption": policy_logdev[:, 11 * n + 2],
            "Agg. Labor": policy_logdev[:, 11 * n + 3],
            "Agg. Capital": policy_logdev[:, 11 * n + 6],
            "Agg. Output": policy_logdev[:, 11 * n + 4],
            "Agg. GDP": policy_logdev[:, 11 * n + 4],
            "Agg. Investment": policy_logdev[:, 11 * n + 5],
            "Intratemporal Utility": policy_logdev[:, 11 * n + 7],
        }

    return reaggregate_aggregates(
        state_logdev=simul[: 2 * n, :].T,
        policies_logdev=policy_logdev,
        policies_ss=policies_ss,
        state_ss=state_ss,
        log_policy_count=11 * n + 7,
        utility_intratemp_idx=11 * n + 7,
        P_weights=jnp.log(P_ergodic) - policies_ss[8 * n : 9 * n],
        Pk_weights=jnp.log(Pk_ergodic) - policies_ss[2 * n : 3 * n],
    )


def compute_model_moments_with_consistent_aggregation(
    simul_obs: jnp.ndarray,
    simul_policies: jnp.ndarray,
    policies_ss: jnp.ndarray,
    state_ss: jnp.ndarray,
    P_ergodic: jnp.ndarray,
    Pk_ergodic: jnp.ndarray,
    n_sectors: int,
    ergodic_price_aggregation: bool = False,
) -> dict:
    """
    Compute MATLAB-style `ModelStats` moments for a nonlinear simulation path.

    Aggregate moments use the same fixed-price aggregation as the Python analysis
    pipeline, while sectoral/comovement moments mirror MATLAB's stored
    `ModelStats` definitions.
    """
    n = n_sectors

    obs_np = np.asarray(simul_obs, dtype=float)
    policies_np = np.asarray(simul_policies, dtype=float)
    policies_ss_np = np.asarray(policies_ss, dtype=float)
    state_ss_np = np.asarray(state_ss, dtype=float)
    P_ergodic_np = np.asarray(P_ergodic, dtype=float)
    Pk_ergodic_np = np.asarray(Pk_ergodic, dtype=float)

    policies_ss_levels = np.exp(policies_ss_np)
    state_ss_levels = np.exp(state_ss_np[:n])

    C_ss = policies_ss_levels[:n]
    L_ss = policies_ss_levels[n : 2 * n]
    Pk_ss = policies_ss_levels[2 * n : 3 * n]
    Mout_ss = policies_ss_levels[5 * n : 6 * n]
    I_ss = policies_ss_levels[6 * n : 7 * n]
    Iout_ss = policies_ss_levels[7 * n : 8 * n]
    Q_ss = policies_ss_levels[9 * n : 10 * n]
    P_ss = policies_ss_levels[8 * n : 9 * n]

    C_levels = C_ss[None, :] * np.exp(policies_np[:, :n])
    L_levels = L_ss[None, :] * np.exp(policies_np[:, n : 2 * n])
    Mout_levels = Mout_ss[None, :] * np.exp(policies_np[:, 5 * n : 6 * n])
    Iout_levels = Iout_ss[None, :] * np.exp(policies_np[:, 7 * n : 8 * n])
    Q_levels = Q_ss[None, :] * np.exp(policies_np[:, 9 * n : 10 * n])
    K_levels = state_ss_levels[None, :] * np.exp(obs_np[:, :n])

    c_logdev = policies_np[:, :n].T
    l_logdev = policies_np[:, n : 2 * n].T
    i_logdev = policies_np[:, 6 * n : 7 * n].T
    a_logdev = obs_np[:, n : 2 * n].T
    q_logdev = policies_np[:, 9 * n : 10 * n].T

    Cagg_exp = C_levels @ P_ergodic_np
    Iagg_exp = Iout_levels @ P_ergodic_np
    GDPagg_exp = (Q_levels - Mout_levels) @ P_ergodic_np
    Kagg = K_levels @ Pk_ergodic_np
    L_hc = np.sum(L_levels, axis=1)

    Cagg_exp_ss = np.sum(P_ergodic_np * C_ss)
    Iagg_exp_ss = np.sum(P_ergodic_np * Iout_ss)
    GDPagg_exp_ss = np.sum(P_ergodic_np * (Q_ss - Mout_ss))
    Kagg_ss = state_ss_levels @ Pk_ergodic_np
    L_hc_ss = np.sum(L_ss)
    Cagg_policy_ss = float(np.exp(policies_ss_np[11 * n + 2]))
    GDPagg_policy_ss = float(np.exp(policies_ss_np[11 * n + 4]))
    Iagg_policy_ss = float(np.exp(policies_ss_np[11 * n + 5]))

    eps = 1e-12
    if ergodic_price_aggregation:
        C_logdev = np.log(np.maximum(Cagg_exp, eps)) - np.log(np.maximum(Cagg_exp_ss, eps))
        I_logdev = np.log(np.maximum(Iagg_exp, eps)) - np.log(np.maximum(Iagg_exp_ss, eps))
        GDP_logdev = np.log(np.maximum(GDPagg_exp, eps)) - np.log(np.maximum(GDPagg_exp_ss, eps))
        K_logdev = np.log(np.maximum(Kagg, eps)) - np.log(np.maximum(Kagg_ss, eps))
        L_hc_logdev = np.log(np.maximum(L_hc, eps)) - np.log(np.maximum(L_hc_ss, eps))
        share_c_denominator = GDPagg_exp_ss
        share_i_denominator = GDPagg_exp_ss
        share_c_numerator = Cagg_exp_ss
        share_i_numerator = Iagg_exp_ss
    else:
        C_logdev = policies_np[:, 11 * n + 2]
        L_hc_logdev = policies_np[:, 11 * n + 3]
        GDP_logdev = policies_np[:, 11 * n + 4]
        I_logdev = policies_np[:, 11 * n + 5]
        K_logdev = policies_np[:, 11 * n + 6]
        share_c_denominator = GDPagg_policy_ss
        share_i_denominator = GDPagg_policy_ss
        share_c_numerator = Cagg_policy_ss
        share_i_numerator = Iagg_policy_ss

    va_price_weights = P_ergodic_np if ergodic_price_aggregation else P_ss
    va_sector_ss = va_price_weights * (Q_ss - Mout_ss)
    va_sector_levels = va_price_weights[None, :] * (Q_levels - Mout_levels)
    va_sector_logdev = np.log(np.maximum(va_sector_levels, eps)) - np.log(np.maximum(va_sector_ss[None, :], eps))
    va_weights = va_sector_ss / np.sum(va_sector_ss)
    go_weights = Q_ss / np.sum(Q_ss)
    emp_weights = L_ss / np.sum(L_ss)
    inv_weights = (I_ss * Pk_ss) / np.sum(I_ss * Pk_ss)

    aggregate_moments = {
        "C": _compute_univariate_moments(C_logdev),
        "I": _compute_univariate_moments(I_logdev),
        "GDP": _compute_univariate_moments(GDP_logdev),
        "L": _compute_univariate_moments(L_hc_logdev),
        "K": _compute_univariate_moments(K_logdev),
    }

    corr_matrix_C, avg_pairwise_corr_C = _safe_corr_matrix_rows(c_logdev)
    corr_matrix_VA, avg_pairwise_corr_VA = _safe_corr_matrix_rows(va_sector_logdev.T)
    corr_matrix_L, avg_pairwise_corr_L = _safe_corr_matrix_rows(l_logdev)
    corr_matrix_I, avg_pairwise_corr_I = _safe_corr_matrix_rows(i_logdev)

    sigma_VA_sectoral = np.std(va_sector_logdev, axis=0)
    sigma_L_sectoral = np.std(l_logdev, axis=1)
    sigma_I_sectoral = np.std(i_logdev, axis=1)

    domar_simul = q_logdev - GDP_logdev[None, :]
    sigma_Domar_sectoral = np.std(domar_simul, axis=1)

    A_VA_logdev = va_weights @ a_logdev
    omega_Q = (va_price_weights * Q_ss) / np.sum(va_price_weights * (Q_ss - Mout_ss))
    A_GO_logdev = omega_Q @ a_logdev
    corr_L_TFP_sectoral = np.array([_safe_corr(l_logdev[j], a_logdev[j]) for j in range(n)])

    return {
        "sigma_VA_agg": float(np.std(GDP_logdev)),
        "sigma_C_agg": float(np.std(C_logdev)),
        "sigma_L_agg": float(np.std(L_hc_logdev)),
        "sigma_L_hc_agg": float(np.std(L_hc_logdev)),
        "sigma_I_agg": float(np.std(I_logdev)),
        "sigma_K_agg": float(np.std(K_logdev)),
        "aggregate_definition": (
            "ergodic_price_reaggregation" if ergodic_price_aggregation else "model_implied_policy_aggregates"
        ),
        "sample_window": "shocks_simul",
        "aggregate_moments": aggregate_moments,
        "share_C": float(share_c_numerator / share_c_denominator),
        "share_I": float(share_i_numerator / share_i_denominator),
        "corr_CI_agg": _safe_corr(C_logdev, I_logdev),
        "corr_L_C_agg": _safe_corr(L_hc_logdev, C_logdev),
        "corr_I_C_agg": _safe_corr(I_logdev, C_logdev),
        "rho_VA_agg": _safe_corr(GDP_logdev[:-1], GDP_logdev[1:]) if GDP_logdev.size >= 2 else float("nan"),
        "avg_pairwise_corr_C": avg_pairwise_corr_C,
        "avg_pairwise_corr_VA": avg_pairwise_corr_VA,
        "avg_pairwise_corr_L": avg_pairwise_corr_L,
        "avg_pairwise_corr_I": avg_pairwise_corr_I,
        "sigma_VA_avg": float(np.sum(va_weights * sigma_VA_sectoral)),
        "sigma_VA_sectoral": sigma_VA_sectoral,
        "sigma_L_avg": float(np.sum(va_weights * sigma_L_sectoral)),
        "sigma_I_avg": float(np.sum(va_weights * sigma_I_sectoral)),
        "sigma_L_avg_empweighted": float(np.sum(emp_weights * sigma_L_sectoral)),
        "sigma_I_avg_invweighted": float(np.sum(inv_weights * sigma_I_sectoral)),
        "sigma_Domar_avg": float(np.sum(go_weights * sigma_Domar_sectoral)),
        "corr_matrix_C": corr_matrix_C,
        "sigma_L_sectoral": sigma_L_sectoral,
        "sigma_I_sectoral": sigma_I_sectoral,
        "sigma_Domar_sectoral": sigma_Domar_sectoral,
        "corr_matrix_VA": corr_matrix_VA,
        "corr_matrix_L": corr_matrix_L,
        "corr_matrix_I": corr_matrix_I,
        "corr_L_TFP_agg": _safe_corr(L_hc_logdev, A_VA_logdev),
        "corr_L_TFP_GO_agg": _safe_corr(L_hc_logdev, A_GO_logdev),
        "corr_L_TFP_sectoral": corr_L_TFP_sectoral,
        "corr_L_TFP_sectoral_avg_vashare": _weighted_mean_ignore_nan(corr_L_TFP_sectoral, va_weights),
        "corr_L_TFP_sectoral_avg_empshare": _weighted_mean_ignore_nan(corr_L_TFP_sectoral, emp_weights),
        "va_weights": va_weights,
        "go_weights": go_weights,
        "emp_weights": emp_weights,
        "inv_weights": inv_weights,
    }


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
        "c_util": 13 * n,
        "l_util": 13 * n + 1,
        "c_agg": 13 * n + 2,
        "l_agg": 13 * n + 3,
        "gdp_agg": 13 * n + 4,
        "i_agg": 13 * n + 5,
        "k_agg": 13 * n + 6,
        "utility_intratemp": 13 * n + 7,
    }


def _compute_univariate_moments(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    moments = {"mean": float("nan"), "std": float("nan"), "skewness": float("nan"), "kurtosis": float("nan")}
    if x.size == 0:
        return moments

    mu = float(np.mean(x))
    sigma = float(np.std(x))
    moments["mean"] = mu
    moments["std"] = sigma
    if sigma == 0:
        return moments

    z = (x - mu) / sigma
    moments["skewness"] = float(np.mean(z**3))
    moments["kurtosis"] = float(np.mean(z**4))
    return moments


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def _weighted_mean_ignore_nan(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan")

    values = values[mask]
    weights = weights[mask]
    weights = weights / np.sum(weights)
    return float(np.sum(weights * values))


def _safe_corr_matrix_rows(data: np.ndarray) -> tuple[np.ndarray, float]:
    data = np.asarray(data, dtype=float)
    n_rows = data.shape[0]
    corr_matrix = np.full((n_rows, n_rows), np.nan, dtype=float)
    for i in range(n_rows):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n_rows):
            rho = _safe_corr(data[i], data[j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    upper = np.triu(np.ones((n_rows, n_rows), dtype=bool), 1)
    values = corr_matrix[upper]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return corr_matrix, float("nan")
    return corr_matrix, float(np.mean(values))


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
                                  sigma_I_agg
                   Optional: rho_VA_agg (autocorrelation), var_cov_agg (correlations)
        n_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with analysis variables (log deviations from SS):
            - "Agg. Consumption": samples of log(C/C_ss)
            - "Agg. Labor": samples of log(L/L_ss)
            - "Agg. Output": samples of log(Y/Y_ss)
            - "Agg. Investment": samples of log(I/I_ss)
    """
    key = random.PRNGKey(seed)

    # Map theo_stats keys to analysis variable names
    # Log-linear: log(X/X_ss) ~ N(0, sigma²)
    var_mapping = {
        "Agg. Consumption": "sigma_C_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. GDP": "sigma_VA_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Capital": "sigma_K_agg",
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
                   Expected keys: sigma_C_agg, sigma_L_agg, sigma_VA_agg, sigma_I_agg, sigma_K_agg
        label: Label for this experiment in the output

    Returns:
        Dictionary in format {label: {var_label: {"Mean": val, "Sd": val, "Skewness": val, "Excess Kurtosis": val}}}
        Ready to pass to create_descriptive_stats_table as theoretical_stats
    """
    var_mapping = {
        "Agg. Consumption": "sigma_C_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. GDP": "sigma_VA_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Capital": "sigma_K_agg",
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
    are taken from it when available (sigma_C_agg, sigma_VA_agg, sigma_L_agg, sigma_I_agg, sigma_K_agg).
    When model_stats is provided, all main aggregate volatilities should come from it.

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
                     When present, used for sigma_C_agg, sigma_VA_agg, sigma_L_agg, sigma_I_agg, sigma_K_agg.
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
        "Agg. Consumption": "sigma_C_agg",
        "Agg. GDP": "sigma_VA_agg",
        "Agg. Labor": "sigma_L_agg",
        "Agg. Investment": "sigma_I_agg",
        "Agg. Capital": "sigma_K_agg",
    }

    has_new_format = "policies_std" in determ_stats or "policies_mean" in determ_stats

    if has_new_format:
        agg_indices = {
            "Agg. Consumption": 11 * n + 2,
            "Agg. Labor": 11 * n + 3,
            "Agg. GDP": 11 * n + 4,
            "Agg. Investment": 11 * n + 5,
            "Agg. Capital": 11 * n + 6,
            "Intratemporal Utility": 11 * n + 7,
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
            "Agg. GDP": ("Yagg_mean_logdev", "Yagg_volatility", "Yagg_skewness", "Yagg_kurtosis"),
            "Agg. Investment": ("Iagg_mean_logdev", "Iagg_volatility", "Iagg_skewness", "Iagg_kurtosis"),
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
