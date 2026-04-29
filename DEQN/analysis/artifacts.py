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


def _write_aggregate_series(path: str, variables: dict[str, Any]) -> None:
    arrays = {label: np.asarray(values).reshape(-1) for label, values in variables.items()}
    if not arrays:
        return
    max_len = max(values.shape[0] for values in arrays.values())
    fieldnames = ["period", *arrays.keys()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(max_len):
            row = {"period": idx}
            for label, values in arrays.items():
                row[label] = float(values[idx]) if idx < values.shape[0] else ""
            writer.writerow(row)
    print(f"Saved: {path}")


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
            npz_path = os.path.join(artifacts_dir, f"simulation_{safe_method}_{analysis_name}.npz")
            np.savez_compressed(npz_path, **arrays)
            print(f"Saved: {npz_path}")

    for method, variables in analysis_variables_data.items():
        aggregates_path = os.path.join(simulation_dir, f"aggregates_{_safe_name(method)}_{analysis_name}.csv")
        _write_aggregate_series(aggregates_path, variables)

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
