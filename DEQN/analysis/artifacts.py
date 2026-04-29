import csv
import json
import os
from typing import Any

import numpy as np

from DEQN.analysis.summary_stats import compute_descriptive_stats_rows, compute_stochastic_ss_rows


def _safe_name(label: str) -> str:
    return (
        str(label)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("-", "_")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(np.asarray(value))
    except Exception:
        return str(value)


def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"Saved: {path}")


def _array_mean_sd(values: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return np.asarray([float(np.nanmean(arr))]), np.asarray([float(np.nanstd(arr, ddof=1))])
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)


def _write_array_moments(path: str, arrays: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for group_name, values in arrays.items():
        mean_values, sd_values = _array_mean_sd(values)
        for idx, (mean_value, sd_value) in enumerate(zip(mean_values.ravel(), sd_values.ravel())):
            rows.append(
                {
                    "group": group_name,
                    "index": idx,
                    "mean": float(mean_value),
                    "sd": float(sd_value),
                }
            )
    _write_csv(path, rows, ["group", "index", "mean", "sd"])


def save_analysis_artifacts(
    *,
    analysis_dir: str,
    simulation_dir: str,
    analysis_name: str,
    config: dict[str, Any],
    raw_simulation_data: dict[str, dict[str, Any]],
    analysis_variables_data: dict[str, dict[str, Any]],
    welfare_costs: dict[str, Any],
    stochastic_ss_states: dict[str, Any],
    stochastic_ss_policies: dict[str, Any],
    stochastic_ss_data: dict[str, dict[str, Any]],
) -> dict[str, str]:
    artifacts_dir = os.path.join(analysis_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    metadata_path = os.path.join(artifacts_dir, f"analysis_artifact_{analysis_name}.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(
            {
                "analysis_name": analysis_name,
                "config": _json_safe(config),
                "methods": list(raw_simulation_data.keys()),
                "analysis_variable_methods": list(analysis_variables_data.keys()),
                "stochastic_ss_methods": list(stochastic_ss_data.keys()),
            },
            metadata_file,
            indent=2,
        )
    print(f"Saved: {metadata_path}")

    for method, sim_data in raw_simulation_data.items():
        safe_method = _safe_name(method)
        arrays = {}
        if sim_data.get("simul_obs") is not None:
            arrays["states"] = np.asarray(sim_data["simul_obs"])
        if sim_data.get("simul_policies") is not None:
            arrays["policies"] = np.asarray(sim_data["simul_policies"])
        if arrays:
            moments_path = os.path.join(artifacts_dir, f"simulation_moments_{safe_method}_{analysis_name}.csv")
            _write_array_moments(moments_path, arrays)

    welfare_path = os.path.join(analysis_dir, f"welfare_{analysis_name}.csv")
    welfare_rows = [
        {"method": method, "welfare_cost": float(np.asarray(value))}
        for method, value in welfare_costs.items()
    ]
    _write_csv(welfare_path, welfare_rows, ["method", "welfare_cost"])

    descriptive_rows = compute_descriptive_stats_rows(
        raw_simulation_data=raw_simulation_data,
        analysis_variables_data=analysis_variables_data,
    )
    descriptive_path = os.path.join(simulation_dir, f"descriptive_stats_{analysis_name}.csv")
    _write_csv(
        descriptive_path,
        descriptive_rows,
        ["method", "variable_group", "variable", "mean", "sd", "skewness", "excess_kurtosis", "n"],
    )

    stochss_rows = compute_stochastic_ss_rows(
        stochastic_ss_states=stochastic_ss_states,
        stochastic_ss_policies=stochastic_ss_policies,
        stochastic_ss_data=stochastic_ss_data,
    )
    stochss_path = os.path.join(analysis_dir, f"stochastic_ss_{analysis_name}.csv")
    _write_csv(stochss_path, stochss_rows, ["method", "variable_group", "variable", "value", "value_percent"])

    return {
        "metadata": metadata_path,
        "welfare": welfare_path,
        "descriptive_stats": descriptive_path,
        "stochastic_ss": stochss_path,
    }
