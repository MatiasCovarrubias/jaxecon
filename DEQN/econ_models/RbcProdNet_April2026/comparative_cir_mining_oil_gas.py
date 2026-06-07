#!/usr/bin/env python3
"""
Compare sector-level CIR measures across saved RbcProdNet analyses.

Run after each single experiment has been processed by DEQN.analysis with a
version that writes IRs/IR_tables/cir_sector_values_<analysis_name>.csv.
"""

import csv
import os
import sys
from typing import Any

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    repo_root = "/content/jaxecon"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = "/content/drive/MyDrive/Jaxecontemp"
else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

from DEQN.analysis.report_specs import escape_latex  # noqa: E402


config = {
    "model_dir": "RbcProdNet_April2026",
    "comparative_name": "mining_oil_gas_scaled_shock_May2026_moreshocks",
    "sector_idx": 0,
    "sector_label": "Mining, Oil and Gas",
    "analyses": [
        {
            "analysis_name": "benchmark_May2026_moreshocks",
            "label": "Baseline",
        },
        {
            "analysis_name": "scaled_shock_May2026_moreshocks",
            "label": "Scaled volatility",
        },
    ],
}

CIR_MEASURE_PANELS = [
    (
        "Global solution vs MIT shock",
        ["Opt. atten. (-)", "Opt. atten. (+)", "Global asym."],
    ),
]

CIR_MEASURE_FORMULAS = {
    "Opt. atten. (-)": r"$100\times(1-GIR_{GS}(-)/GIR_{MIT}(-))$",
    "Opt. atten. (+)": r"$100\times(1-GIR_{GS}(+)/GIR_{MIT}(+))$",
    "Global asym.": r"$100\times(1-|GIR_{GS}(-)/GIR_{GS}(+)|)$",
}


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required sector CIR artifact not found: {path}\n"
            "Rerun DEQN.analysis for this analysis after pulling the sector CIR export change."
        )
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"Saved: {path}")


def _latex_label_token(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(text))


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _analysis_dir(model_dir: str, analysis_name: str) -> str:
    return os.path.join(model_dir, "analysis", analysis_name)


def _cir_sector_csv_path(model_dir: str, analysis_name: str) -> str:
    return os.path.join(
        _analysis_dir(model_dir, analysis_name),
        "IRs",
        "IR_tables",
        f"cir_sector_values_{analysis_name}.csv",
    )


def _load_sector_rows(model_dir: str, analysis_spec: dict[str, str], sector_idx: int) -> list[dict[str, Any]]:
    analysis_name = analysis_spec["analysis_name"]
    analysis_label = analysis_spec.get("label") or analysis_name
    rows = []
    for row in _read_csv_rows(_cir_sector_csv_path(model_dir, analysis_name)):
        if str(row.get("sector_idx")) != str(sector_idx):
            continue
        if row.get("measure") not in CIR_MEASURE_FORMULAS:
            continue
        rows.append(
            {
                "analysis_name": analysis_name,
                "analysis_label": analysis_label,
                "sector_idx": row.get("sector_idx", ""),
                "sector_label": row.get("sector_label", ""),
                "shock_size": row.get("shock_size", ""),
                "measure_panel": row.get("measure_panel", ""),
                "measure": row.get("measure", ""),
                "formula": CIR_MEASURE_FORMULAS.get(row.get("measure", ""), ""),
                "value": row.get("value", ""),
                "value_percent": row.get("value_percent", ""),
            }
        )
    if not rows:
        raise ValueError(f"No sector {sector_idx} CIR rows found for analysis '{analysis_name}'.")
    return rows


def _load_all_sector_rows(model_dir: str, analysis_specs: list[dict[str, str]], sector_idx: int) -> list[dict[str, Any]]:
    if len(analysis_specs) != 2:
        raise ValueError("This comparison table expects exactly two analyses: baseline and scaled experiment.")
    rows = []
    for analysis_spec in analysis_specs:
        rows.extend(_load_sector_rows(model_dir, analysis_spec, sector_idx))
    return rows


def _shock_sort_key(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _value_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    return {
        (row["analysis_label"], row["measure"], row["shock_size"]): row.get("value_percent", "")
        for row in rows
    }


def _difference_rows(rows: list[dict[str, Any]], baseline_label: str, experiment_label: str) -> list[dict[str, Any]]:
    lookup = _value_lookup(rows)
    shock_sizes = sorted({row["shock_size"] for row in rows}, key=_shock_sort_key)
    differences = []
    for panel_title, measures in CIR_MEASURE_PANELS:
        for measure in measures:
            for shock_size in shock_sizes:
                baseline_value = lookup.get((baseline_label, measure, shock_size), "")
                experiment_value = lookup.get((experiment_label, measure, shock_size), "")
                try:
                    difference = float(experiment_value) - float(baseline_value)
                except (TypeError, ValueError):
                    difference = ""
                differences.append(
                    {
                        "analysis_label": f"{experiment_label} - {baseline_label}",
                        "measure_panel": panel_title,
                        "measure": measure,
                        "shock_size": shock_size,
                        "value_percent": difference,
                    }
                )
    return differences


def _write_comparison_table_tex(
    path: str,
    *,
    rows: list[dict[str, Any]],
    baseline_label: str,
    experiment_label: str,
    sector_label: str,
    comparative_name: str,
) -> str:
    shock_sizes = sorted({row["shock_size"] for row in rows}, key=_shock_sort_key)
    lookup = _value_lookup(rows)
    difference_lookup = {
        (row["measure"], row["shock_size"]): row.get("value_percent", "")
        for row in _difference_rows(rows, baseline_label, experiment_label)
    }
    analysis_labels = [baseline_label, experiment_label, f"{experiment_label} - {baseline_label}"]
    column_count = 3 + len(shock_sizes)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Sector-level cumulative impulse-response comparison: {escape_latex(sector_label)}}}",
        rf"\label{{tab:cir_sector_comparison_{_latex_label_token(comparative_name)}}}",
        r"\begin{tabular}{lll" + "r" * len(shock_sizes) + "}",
        r"\toprule",
        "Metric & Analysis & Formula & "
        + " & ".join(f"{escape_latex(f'{float(shock_size):g}%')}" for shock_size in shock_sizes)
        + r" \\",
        r"\midrule",
    ]

    for panel_title, measures in CIR_MEASURE_PANELS:
        lines.append(rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{{escape_latex(panel_title)}}}}} \\")
        for measure in measures:
            for row_idx, analysis_label in enumerate(analysis_labels):
                values = []
                for shock_size in shock_sizes:
                    if analysis_label == analysis_labels[-1]:
                        values.append(_format_float(difference_lookup.get((measure, shock_size), "")))
                    else:
                        values.append(_format_float(lookup.get((analysis_label, measure, shock_size), "")))
                metric_cell = escape_latex(measure) if row_idx == 0 else ""
                formula_cell = CIR_MEASURE_FORMULAS[measure] if row_idx == 0 else ""
                lines.append(
                    f"{metric_cell} & {escape_latex(analysis_label)} & {formula_cell} & "
                    + " & ".join(values)
                    + r" \\"
                )
        lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} The table reports volatility-sensitive CIR measures for the selected sector only. "
            r"CIR is the sum over the displayed IR horizon of the aggregate consumption response to a "
            r"sectoral TFP shock. Values are reported in percent; the difference row is measured in "
            r"percentage points. $GIR_{GS}$ is the global-solution CIR and $GIR_{MIT}$ is the MIT shock CIR.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as tex_file:
        tex_file.write("\n".join(lines))
    print(f"Saved: {path}")
    return path


def _write_wrapper(output_dir: str, comparative_name: str, table_path: str) -> str:
    wrapper_path = os.path.join(output_dir, f"comparative_{comparative_name}.tex")
    rel_path = os.path.relpath(table_path, output_dir).replace(os.sep, "/")
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\floatplacement{table}{H}",
        r"\begin{document}",
        rf"\input{{{rel_path}}}",
        r"\end{document}",
        "",
    ]
    with open(wrapper_path, "w") as wrapper_file:
        wrapper_file.write("\n".join(lines))
    print(f"Saved: {wrapper_path}")
    return wrapper_path


def main():
    model_dir = os.path.join(base_dir, config["model_dir"])
    output_dir = os.path.join(model_dir, "analysis", "comparisons", config["comparative_name"])
    os.makedirs(output_dir, exist_ok=True)

    analysis_specs = config["analyses"]
    baseline_label = analysis_specs[0].get("label") or analysis_specs[0]["analysis_name"]
    experiment_label = analysis_specs[1].get("label") or analysis_specs[1]["analysis_name"]
    sector_idx = int(config["sector_idx"])

    rows = _load_all_sector_rows(model_dir, analysis_specs, sector_idx)
    sector_label = config.get("sector_label") or rows[0].get("sector_label") or f"Sector {sector_idx + 1}"
    comparison_rows = rows + _difference_rows(rows, baseline_label, experiment_label)

    comparison_csv = os.path.join(output_dir, f"cir_sector_comparison_{config['comparative_name']}.csv")
    _write_csv(
        comparison_csv,
        comparison_rows,
        [
            "analysis_name",
            "analysis_label",
            "sector_idx",
            "sector_label",
            "shock_size",
            "measure_panel",
            "measure",
            "formula",
            "value",
            "value_percent",
        ],
    )

    table_path = _write_comparison_table_tex(
        os.path.join(output_dir, f"cir_sector_comparison_{config['comparative_name']}.tex"),
        rows=rows,
        baseline_label=baseline_label,
        experiment_label=experiment_label,
        sector_label=sector_label,
        comparative_name=config["comparative_name"],
    )
    _write_wrapper(output_dir, config["comparative_name"], table_path)

    print("\nSector CIR comparison complete.")
    return {
        "rows": comparison_rows,
        "comparison_csv": comparison_csv,
        "table": table_path,
    }


if __name__ == "__main__":
    main()
