#!/usr/bin/env python3
"""
Grid simulation analysis functions for RBC Production Network model.

This module contains functions for running seed-length grid diagnostics
to assess simulation quality, convergence, and stationarity properties
across all model variables (states, policies, and aggregates).

The analysis is general and computes diagnostics for all variables returned
by the economic model's get_aggregates function, rather than focusing on
specific measures like welfare or sectoral capital.
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from DEQN.analysis.simul_analysis import create_episode_simulation_fn_verbose


def _estimate_iact_batch_means(x: jnp.ndarray, num_batches: int = 20) -> float:
    """
    Batch-means estimator for integrated autocorrelation time (IACT).

    x: 1D array (time series)
    Returns: tau_hat >= 1
    """
    x = jnp.asarray(x)
    n = x.shape[0]
    if n < num_batches * 2:
        return float(1.0)
    batch_size = n // num_batches
    n_eff = batch_size * num_batches
    x_used = x[:n_eff]
    x_reshaped = x_used.reshape((num_batches, batch_size))
    batch_means = jnp.mean(x_reshaped, axis=1)
    var_x = jnp.var(x_used, ddof=1)
    var_b = jnp.var(batch_means, ddof=1)
    tau_hat = jnp.where(var_x > 0, batch_size * var_b / var_x, 1.0)
    tau_hat = jnp.maximum(tau_hat, 1.0)
    return float(tau_hat)


def _ols_slope_per_period(y: jnp.ndarray) -> float:
    """
    OLS slope of y_t on t (t = 0,1,...,T-1). Returns slope per period.
    """
    y = jnp.asarray(y)
    t = jnp.arange(y.shape[0], dtype=y.dtype)
    t_mean = jnp.mean(t)
    y_mean = jnp.mean(y)
    cov_ty = jnp.mean((t - t_mean) * (y - y_mean))
    var_t = jnp.mean((t - t_mean) ** 2)
    slope = jnp.where(var_t > 0, cov_ty / var_t, 0.0)
    return float(slope)


def _ood_fraction(
    states: jnp.ndarray,
    policies: jnp.ndarray,
    state_sd: jnp.ndarray,
    policies_sd: jnp.ndarray,
    thresholds=(2.0, 3.0, 4.0),
) -> dict:
    """
    Compute fraction of normalized |deviations| exceeding sigma thresholds.

    Args:
        states: (T, n_states) - log deviations of states
        policies: (T, n_policies) - log deviations of policies
        state_sd: (n_states,) - standard deviations for state normalization
        policies_sd: (n_policies,) - standard deviations for policy normalization
        thresholds: sigma levels to test (e.g., 2, 3, 4 sigmas)

    Returns: dict {thr: fraction}
    """
    # Normalize states and policies by their standard deviations
    states_normalized = states / state_sd
    policies_normalized = policies / policies_sd

    # Combine all normalized deviations
    all_normalized = jnp.concatenate([states_normalized.flatten(), policies_normalized.flatten()])
    all_normalized_abs = jnp.abs(all_normalized)

    res = {}
    total = all_normalized_abs.size
    for thr in thresholds:
        frac = jnp.sum(all_normalized_abs > thr) / total
        res[float(thr)] = float(frac)
    return res


def _shock_diagnostics(simul_state: jnp.ndarray, econ_model) -> dict:
    """
    Recover shocks from state process: a_{t+1} = rho * a_t + eps_t.
    Returns mean and covariance diagnostics of eps_t.
    """
    nS = econ_model.n_sectors
    # Denormalize states to get a_t
    st_loglevel = simul_state + econ_model.state_ss
    a_t = st_loglevel[:, nS:]
    eps = a_t[1:, :] - econ_model.rho * a_t[:-1, :]
    mean_eps = jnp.mean(eps, axis=0)
    cov_eps = jnp.cov(eps.T)
    # Norm diagnostics
    sigma = econ_model.Sigma_A
    num = jnp.linalg.norm(cov_eps - sigma)
    den = jnp.linalg.norm(sigma) + 1e-12
    ratio = num / den
    return {
        "mean_eps_norm": float(jnp.linalg.norm(mean_eps)),
        "cov_diff_rel_fro": float(ratio),
    }


def _compute_fixed_weights(econ_model, precision) -> tuple:
    """
    Return zero log-deviation weights so that levels are fixed at steady-state prices.
    """
    zeros = jnp.zeros((econ_model.n_sectors,), dtype=precision)
    return zeros, zeros, zeros


def run_seed_length_grid(
    econ_model,
    train_state,
    base_config: dict,
    lengths: list,
    n_seeds: int,
    burnin_fracs: list,
    iact_num_batches: int,
):
    """
    Run a grid over episode lengths and seeds, computing comprehensive diagnostics
    for all model variables (states, policies, aggregates) to distinguish sampling
    error from drift and out-of-distribution behavior.

    Uses fixed steady-state weights for aggregate computation to ensure stationarity.
    Computes IACT, linear trends, cross-seed dispersion, scaling relationships,
    and sigma-normalized out-of-distribution fractions (at 2σ, 3σ, 4σ levels)
    for all variables rather than focusing on specific measures.

    Args:
        econ_model: Economic model instance with get_aggregates method
        train_state: Trained neural network state
        base_config: Base configuration dictionary
        lengths: List of episode lengths to test
        n_seeds: Number of random seeds per length
        burnin_fracs: List of burn-in fractions
        iact_num_batches: Number of batches for IACT estimation

    Returns:
        dict: Comprehensive results with diagnostics for states, policies, and aggregates
    """
    results = {}
    precision = jnp.float64 if base_config["double_precision"] else jnp.float32
    fixed_P, fixed_Pk, fixed_Pm = _compute_fixed_weights(econ_model, precision)

    for T in lengths:
        results[T] = {}
        for bfrac in burnin_fracs:
            burn = int(max(1, round(bfrac * T)))
            print(f"  Grid run: T={T}, burn-in={burn} ({bfrac:.0%})")

            # Build per-length config and simulation function
            cfg = dict(base_config)
            cfg["periods_per_epis"] = T
            cfg["burn_in_periods"] = burn
            simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, cfg))

            per_seed = []

            for s in range(n_seeds):
                episode_rng = random.PRNGKey(int(1_000_000 + s))
                simul_state, simul_policies = simulation_fn(train_state, episode_rng)
                simul_state = simul_state[burn:]
                simul_policies = simul_policies[burn:]

                # Aggregates with fixed weights
                aggs_ts = jax.vmap(econ_model.get_aggregates, in_axes=(0, 0, None, None, None))(
                    simul_state, simul_policies, fixed_P, fixed_Pk, fixed_Pm
                )

                # Compute mean values for all states, policies, and aggregates
                state_mean = jnp.mean(simul_state, axis=0)
                policies_mean = jnp.mean(simul_policies, axis=0)
                aggregates_mean = jnp.mean(aggs_ts, axis=0)

                # IACT and drift for all aggregates
                iact_aggregates = [
                    _estimate_iact_batch_means(aggs_ts[:, i], num_batches=iact_num_batches)
                    for i in range(aggs_ts.shape[1])
                ]
                slope_aggregates = [_ols_slope_per_period(aggs_ts[:, i]) for i in range(aggs_ts.shape[1])]

                # OOD and shocks
                ood = _ood_fraction(simul_state, simul_policies, econ_model.state_sd, econ_model.policies_sd)
                shocks = _shock_diagnostics(simul_state, econ_model)

                per_seed.append(
                    {
                        "state_mean": np.asarray(state_mean).tolist(),
                        "policies_mean": np.asarray(policies_mean).tolist(),
                        "aggregates_mean": np.asarray(aggregates_mean).tolist(),
                        "iact_aggregates": iact_aggregates,
                        "trend_slope_aggregates": slope_aggregates,
                        "ood_fraction": ood,
                        "shocks": shocks,
                    }
                )

            # Aggregate across seeds
            if n_seeds > 1:
                # Collect arrays for each variable type
                state_means = jnp.array([d["state_mean"] for d in per_seed])
                policies_means = jnp.array([d["policies_mean"] for d in per_seed])
                aggregates_means = jnp.array([d["aggregates_mean"] for d in per_seed])

                # Calculate cross-seed standard deviations
                sd_state = jnp.std(state_means, axis=0, ddof=1)
                sd_policies = jnp.std(policies_means, axis=0, ddof=1)
                sd_aggregates = jnp.std(aggregates_means, axis=0, ddof=1)

                # Mean standard deviations (for summary)
                sd_state_mean = float(jnp.mean(sd_state))
                sd_policies_mean = float(jnp.mean(sd_policies))
                sd_aggregates_mean = float(jnp.mean(sd_aggregates))
            else:
                sd_state = jnp.zeros(len(per_seed[0]["state_mean"]))
                sd_policies = jnp.zeros(len(per_seed[0]["policies_mean"]))
                sd_aggregates = jnp.zeros(len(per_seed[0]["aggregates_mean"]))
                sd_state_mean = 0.0
                sd_policies_mean = 0.0
                sd_aggregates_mean = 0.0

            # Average diagnostics across seeds
            n_aggs = len(per_seed[0]["iact_aggregates"])
            mean_iact_aggregates = [float(np.mean([d["iact_aggregates"][i] for d in per_seed])) for i in range(n_aggs)]
            mean_slope_aggregates = [
                float(np.mean([d["trend_slope_aggregates"][i] for d in per_seed])) for i in range(n_aggs)
            ]

            mean_ood = {
                thr: float(np.mean([d["ood_fraction"][thr] for d in per_seed]))
                for thr in per_seed[0]["ood_fraction"].keys()
            }
            mean_shock_mean_norm = float(np.mean([d["shocks"]["mean_eps_norm"] for d in per_seed]))
            mean_shock_cov_diff = float(np.mean([d["shocks"]["cov_diff_rel_fro"] for d in per_seed]))

            results[T][bfrac] = {
                "per_seed": per_seed,
                "sd_state": sd_state.tolist(),
                "sd_policies": sd_policies.tolist(),
                "sd_aggregates": sd_aggregates.tolist(),
                "sd_state_mean": sd_state_mean,
                "sd_policies_mean": sd_policies_mean,
                "sd_aggregates_mean": sd_aggregates_mean,
                "avg_iact_aggregates": mean_iact_aggregates,
                "avg_trend_slope_aggregates": mean_slope_aggregates,
                "avg_ood_fraction": mean_ood,
                "avg_shock_mean_norm": mean_shock_mean_norm,
                "avg_shock_cov_diff": mean_shock_cov_diff,
            }

    # SD vs T slope for all variables (using first burn-in fraction)
    b0 = burnin_fracs[0]
    Ts = sorted(results.keys())
    sd_state_mean = jnp.array([results[T][b0]["sd_state_mean"] for T in Ts])
    sd_policies_mean = jnp.array([results[T][b0]["sd_policies_mean"] for T in Ts])
    sd_aggregates_mean = jnp.array([results[T][b0]["sd_aggregates_mean"] for T in Ts])

    logT = jnp.log(jnp.array(Ts, dtype=jnp.float64))
    log_sd_state = jnp.log(jnp.maximum(sd_state_mean, 1e-12))
    log_sd_policies = jnp.log(jnp.maximum(sd_policies_mean, 1e-12))
    log_sd_aggregates = jnp.log(jnp.maximum(sd_aggregates_mean, 1e-12))

    # Simple slope = Cov/Var
    def _slope(x, y):
        xm = jnp.mean(x)
        ym = jnp.mean(y)
        return float(jnp.mean((x - xm) * (y - ym)) / (jnp.mean((x - xm) ** 2) + 1e-12))

    results["sd_vs_T_slope"] = {
        "state_logsd_logT_slope": _slope(logT, log_sd_state),
        "policies_logsd_logT_slope": _slope(logT, log_sd_policies),
        "aggregates_logsd_logT_slope": _slope(logT, log_sd_aggregates),
    }

    return results


def _print_grid_summary(grid_results: dict):
    """
    Pretty-print key results from run_seed_length_grid.
    """
    print("\nGrid diagnostics summary")
    print("-" * 60)
    slopes = grid_results.get("sd_vs_T_slope", {})
    if slopes:
        state_slope = slopes.get("state_logsd_logT_slope", float("nan"))
        policies_slope = slopes.get("policies_logsd_logT_slope", float("nan"))
        agg_slope = slopes.get("aggregates_logsd_logT_slope", float("nan"))
        print(
            f"SD vs T slope (log-log): state={state_slope:.3f}, policies={policies_slope:.3f}, aggregates={agg_slope:.3f}"
        )

    # List T entries (numeric keys)
    Ts = sorted([k for k in grid_results.keys() if isinstance(k, (int, float))])
    for T in Ts:
        print(f"\n  Length T={int(T)}")
        # For each burn-in fraction
        bkeys = grid_results[T].keys()
        bfracs = sorted([b for b in bkeys if isinstance(b, (int, float))], key=float)
        for b in bfracs:
            r = grid_results[T][b]
            sd_state = r.get("sd_state_mean", float("nan"))
            sd_policies = r.get("sd_policies_mean", float("nan"))
            sd_agg = r.get("sd_aggregates_mean", float("nan"))

            # Get IACT and slope aggregates (show first few for brevity)
            iact_aggs = r.get("avg_iact_aggregates", [])
            slope_aggs = r.get("avg_trend_slope_aggregates", [])

            ood = r.get("avg_ood_fraction", {})
            shock_mean = r.get("avg_shock_mean_norm", float("nan"))
            shock_cov = r.get("avg_shock_cov_diff", float("nan"))

            # Build OOD fraction line in ascending threshold order (now sigma-based)
            if isinstance(ood, dict) and len(ood) > 0:
                thr_sorted = sorted(list(ood.keys()), key=float)
                ood_str = ", ".join([f">{float(th):g}σ: {float(ood[th]):.4f}" for th in thr_sorted])
            else:
                ood_str = "(none)"

            print(
                f"    burn-in={b:.0%} | sd_state={sd_state:.4e} | sd_policies={sd_policies:.4e} | sd_aggregates={sd_agg:.4e}"
            )

            # Show IACT for first few aggregates (C, L, K, Y if available)
            if len(iact_aggs) >= 4:
                print(
                    f"      iact: C={iact_aggs[0]:.2f}, L={iact_aggs[1]:.2f}, K={iact_aggs[2]:.2f}, Y={iact_aggs[3]:.2f}"
                )
            elif len(iact_aggs) > 0:
                iact_str = ", ".join([f"{i:.2f}" for i in iact_aggs[:4]])
                print(f"      iact: {iact_str}")

            # Show trend slopes for first few aggregates
            if len(slope_aggs) >= 4:
                print(
                    f"      trend slopes: C={slope_aggs[0]:.3e}, L={slope_aggs[1]:.3e}, K={slope_aggs[2]:.3e}, Y={slope_aggs[3]:.3e}"
                )
            elif len(slope_aggs) > 0:
                slope_str = ", ".join([f"{s:.3e}" for s in slope_aggs[:4]])
                print(f"      trend slopes: {slope_str}")

            print(
                f"      OOD fractions (σ-normalized): {ood_str} | shock mean norm={shock_mean:.3e}, cov diff (rel Fro)={shock_cov:.3e}"
            )
