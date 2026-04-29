import math
from typing import Any

import numpy as np
from scipy.stats import kurtosis, skew


def _as_1d_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1)


def compute_series_stats(values: Any, *, scale: float = 100.0, min_samples: int = 2) -> dict[str, float] | None:
    arr = _as_1d_float_array(values)
    arr = arr[np.isfinite(arr)]
    if arr.size < min_samples:
        return None

    return {
        "mean": float(np.mean(arr) * scale),
        "sd": float(np.std(arr, ddof=1) * scale) if arr.size > 1 else math.nan,
        "skewness": float(skew(arr)) if arr.size > 2 else math.nan,
        "excess_kurtosis": float(kurtosis(arr)) if arr.size > 3 else math.nan,
        "n": int(arr.size),
    }


def compute_descriptive_stats_rows(
    *,
    raw_simulation_data: dict[str, dict[str, Any]],
    analysis_variables_data: dict[str, dict[str, Any]],
    state_prefix: str = "state",
    policy_prefix: str = "policy",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(method: str, variable_group: str, variable: str, values: Any) -> None:
        stats = compute_series_stats(values)
        if stats is None:
            return
        rows.append(
            {
                "method": method,
                "variable_group": variable_group,
                "variable": variable,
                **stats,
            }
        )

    for method, sim_data in raw_simulation_data.items():
        states = sim_data.get("simul_obs")
        if states is not None:
            states_arr = np.asarray(states)
            if states_arr.ndim == 1:
                add_row(method, "state", f"{state_prefix}_0", states_arr)
            elif states_arr.ndim >= 2:
                for idx in range(states_arr.shape[1]):
                    add_row(method, "state", f"{state_prefix}_{idx}", states_arr[:, idx])

        policies = sim_data.get("simul_policies")
        if policies is not None:
            policies_arr = np.asarray(policies)
            if policies_arr.ndim == 1:
                add_row(method, "policy", f"{policy_prefix}_0", policies_arr)
            elif policies_arr.ndim >= 2:
                for idx in range(policies_arr.shape[1]):
                    add_row(method, "policy", f"{policy_prefix}_{idx}", policies_arr[:, idx])

    for method, variables in analysis_variables_data.items():
        for variable, values in variables.items():
            add_row(method, "aggregate", variable, values)

    return rows


def compute_stochastic_ss_rows(
    *,
    stochastic_ss_states: dict[str, Any],
    stochastic_ss_policies: dict[str, Any],
    stochastic_ss_data: dict[str, dict[str, Any]],
    state_prefix: str = "state",
    policy_prefix: str = "policy",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_vector_rows(method: str, variable_group: str, prefix: str, values: Any) -> None:
        arr = _as_1d_float_array(values)
        for idx, value in enumerate(arr):
            rows.append(
                {
                    "method": method,
                    "variable_group": variable_group,
                    "variable": f"{prefix}_{idx}",
                    "value": float(value),
                    "value_percent": float(value) * 100.0,
                }
            )

    for method, values in stochastic_ss_states.items():
        add_vector_rows(method, "state", state_prefix, values)

    for method, values in stochastic_ss_policies.items():
        add_vector_rows(method, "policy", policy_prefix, values)

    for method, variables in stochastic_ss_data.items():
        for variable, value in variables.items():
            scalar = float(np.asarray(value))
            rows.append(
                {
                    "method": method,
                    "variable_group": "aggregate",
                    "variable": variable,
                    "value": scalar,
                    "value_percent": scalar * 100.0,
                }
            )

    return rows
