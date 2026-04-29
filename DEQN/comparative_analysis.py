#!/usr/bin/env python3
"""
Compare saved single-experiment DEQN analyses.

Run after each single experiment has been processed by DEQN.analysis.
"""

import csv
import json
import os
import sys

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    base_dir = "/content/drive/MyDrive/Jaxecontemp"
else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

from DEQN.analysis.artifacts import _safe_name  # noqa: E402
from DEQN.analysis.report_specs import escape_latex, make_safe_plot_label  # noqa: E402


config = {
    "model_dir": "RbcProdNet_April2026",
    "comparative_name": "comparison",
    "analyses": [
        # {"analysis_name": "GO_shocks_newWDS_v2", "label": "Baseline", "method": "benchmark"},
    ],
    # If omitted, welfare comparison uses each analysis' selected method.
    # Set to "all" to include every welfare row, or to a list of method names.
    "welfare_methods": None,
    "aggregate_variables": [
        "Agg. Consumption",
        "Agg. Investment",
        "Agg. GDP",
        "Agg. Capital",
        "Agg. Labor",
        "Intratemporal Utility",
    ],
    "max_plot_points": 2000,
}


def _read_csv_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required comparative artifact not found: {path}")
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"Saved: {path}")


def _read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _analysis_dir(model_dir, analysis_name):
    return os.path.join(model_dir, "analysis", analysis_name)


def _artifact_path(analysis_dir, analysis_name):
    return os.path.join(analysis_dir, "artifacts", f"analysis_artifact_{analysis_name}.json")


def _load_analysis_spec(model_dir, spec):
    analysis_name = spec["analysis_name"]
    analysis_dir = _analysis_dir(model_dir, analysis_name)
    metadata = _read_json(_artifact_path(analysis_dir, analysis_name))
    if not spec.get("method"):
        raise ValueError(
            f"Comparative analysis requires an explicit method for '{analysis_name}'. "
            "Set each entry as {'analysis_name': ..., 'label': ..., 'method': ...}."
        )
    method = spec["method"]
    label = spec.get("label") or analysis_name
    _validate_method_available(analysis_name, method, metadata)
    return {
        "analysis_name": analysis_name,
        "label": label,
        "method": method,
        "analysis_dir": analysis_dir,
        "simulation_dir": os.path.join(analysis_dir, "simulation"),
        "metadata": metadata,
    }


def _load_all_analyses(model_dir, analysis_specs):
    if not analysis_specs:
        raise ValueError("config['analyses'] must contain at least two saved analyses to compare.")
    analyses = [_load_analysis_spec(model_dir, spec) for spec in analysis_specs]
    if len(analyses) < 2:
        raise ValueError("Comparative analysis requires at least two saved analyses.")
    labels = [analysis["label"] for analysis in analyses]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(f"Comparative analysis labels must be unique. Duplicates: {duplicate_labels}")
    return analyses


def _validate_method_available(analysis_name, method, metadata):
    available_methods = set(metadata.get("analysis_variable_methods") or [])
    available_methods.update(metadata.get("methods") or [])
    available_methods.update(metadata.get("stochastic_ss_methods") or [])
    if method not in available_methods:
        raise ValueError(
            f"Method '{method}' was requested for analysis '{analysis_name}', but it was not found in metadata. "
            f"Available methods: {sorted(available_methods)}"
        )


def _combine_rows(analyses, filename_builder):
    combined = []
    for analysis in analyses:
        path = filename_builder(analysis)
        for row in _read_csv_rows(path):
            combined.append(
                {
                    "analysis_name": analysis["analysis_name"],
                    "analysis_label": analysis["label"],
                    **row,
                }
            )
    return combined


def _filter_welfare_rows(welfare_rows, analyses, welfare_methods):
    if welfare_methods == "all":
        return welfare_rows
    if welfare_methods is None:
        allowed_by_label = {analysis["label"]: {analysis["method"]} for analysis in analyses}
    else:
        configured_methods = {welfare_methods} if isinstance(welfare_methods, str) else set(welfare_methods)
        allowed_by_label = {analysis["label"]: configured_methods for analysis in analyses}
    return [
        row
        for row in welfare_rows
        if row.get("method") in allowed_by_label.get(row.get("analysis_label"), set())
    ]


def _write_pivot_tex(path, *, rows, row_filter, value_column, caption, label):
    analyses = []
    variables = []
    values = {}
    for row in rows:
        if not row_filter(row):
            continue
        analysis_label = row["analysis_label"]
        variable = row["variable"]
        if analysis_label not in analyses:
            analyses.append(analysis_label)
        if variable not in variables:
            variables.append(variable)
        values[(variable, analysis_label)] = row.get(value_column, "")

    if not analyses or not variables:
        return None

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "r" * len(analyses) + "}",
        r"\toprule",
        "Variable & " + " & ".join(rf"\textbf{{{escape_latex(name)}}}" for name in analyses) + r" \\",
        r"\midrule",
    ]
    for variable in variables:
        row_values = [values.get((variable, analysis_label), "") for analysis_label in analyses]
        formatted_values = []
        for value in row_values:
            try:
                formatted_values.append(f"{float(value):.3f}")
            except (TypeError, ValueError):
                formatted_values.append("")
        lines.append(f"{escape_latex(variable)} & " + " & ".join(formatted_values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as tex_file:
        tex_file.write("\n".join(lines))
    print(f"Saved: {path}")
    return path


def _load_aggregate_series(analysis, variable):
    if not analysis["method"]:
        return []
    path = os.path.join(
        analysis["simulation_dir"],
        f"aggregates_{_safe_name(analysis['method'])}_{analysis['analysis_name']}.csv",
    )
    rows = _read_csv_rows(path)
    series = []
    for row in rows:
        if variable not in row or row[variable] == "":
            continue
        try:
            series.append((int(row["period"]), float(row[variable])))
        except ValueError:
            continue
    return series


def _downsample(series, max_points):
    if max_points <= 0 or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series[::step]


def _plot_aggregate_comparisons(analyses, output_dir, variables, max_plot_points):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Matplotlib not available; aggregate comparison figures skipped.")
        return []

    figure_paths = []
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for variable in variables:
        plotted = False
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for analysis in analyses:
            series = _downsample(_load_aggregate_series(analysis, variable), max_plot_points)
            if not series:
                continue
            periods, values = zip(*series)
            ax.plot(periods, values, label=analysis["label"], linewidth=1.5)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(variable)
        ax.set_xlabel("Period")
        ax.set_ylabel("Log deviation")
        ax.legend()
        fig.tight_layout()

        figure_path = os.path.join(figures_dir, f"aggregate_{make_safe_plot_label(variable)}.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {figure_path}")
        figure_paths.append((variable, figure_path))

    return figure_paths


def _write_latex_wrapper(output_dir, comparative_name, table_paths, figure_paths):
    wrapper_path = os.path.join(output_dir, f"comparative_{comparative_name}.tex")
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\floatplacement{table}{H}",
        r"\floatplacement{figure}{H}",
        r"\begin{document}",
    ]
    for table_path in table_paths:
        rel_path = os.path.relpath(table_path, output_dir).replace(os.sep, "/")
        lines.extend([rf"\input{{{rel_path}}}", r"\clearpage"])
    for variable, figure_path in figure_paths:
        rel_path = os.path.relpath(figure_path, output_dir).replace(os.sep, "/")
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.95\textwidth]{{{rel_path}}}",
                rf"\caption{{Comparative aggregate {escape_latex(variable)}.}}",
                r"\end{figure}",
                r"\clearpage",
            ]
        )
    lines.append(r"\end{document}")
    with open(wrapper_path, "w") as wrapper_file:
        wrapper_file.write("\n".join(lines) + "\n")
    print(f"Saved: {wrapper_path}")
    return wrapper_path


def main():
    model_dir = os.path.join(base_dir, config["model_dir"])
    output_dir = os.path.join(model_dir, "analysis", config["comparative_name"])
    os.makedirs(output_dir, exist_ok=True)

    analyses = _load_all_analyses(model_dir, config["analyses"])

    descriptive_rows = _combine_rows(
        analyses,
        lambda analysis: os.path.join(
            analysis["simulation_dir"],
            f"descriptive_stats_{analysis['analysis_name']}.csv",
        ),
    )
    descriptive_csv = os.path.join(output_dir, f"comparative_descriptive_stats_{config['comparative_name']}.csv")
    _write_csv(
        descriptive_csv,
        descriptive_rows,
        [
            "analysis_name",
            "analysis_label",
            "method",
            "variable_group",
            "variable",
            "mean",
            "sd",
            "skewness",
            "excess_kurtosis",
            "n",
        ],
    )

    stochss_rows = _combine_rows(
        analyses,
        lambda analysis: os.path.join(
            analysis["analysis_dir"],
            f"stochastic_ss_{analysis['analysis_name']}.csv",
        ),
    )
    stochss_csv = os.path.join(output_dir, f"comparative_stochastic_ss_{config['comparative_name']}.csv")
    _write_csv(
        stochss_csv,
        stochss_rows,
        ["analysis_name", "analysis_label", "method", "variable_group", "variable", "value", "value_percent"],
    )

    welfare_rows = _combine_rows(
        analyses,
        lambda analysis: os.path.join(
            analysis["analysis_dir"],
            f"welfare_{analysis['analysis_name']}.csv",
        ),
    )
    filtered_welfare_rows = _filter_welfare_rows(welfare_rows, analyses, config.get("welfare_methods"))
    welfare_csv = os.path.join(output_dir, f"comparative_welfare_{config['comparative_name']}.csv")
    _write_csv(welfare_csv, filtered_welfare_rows, ["analysis_name", "analysis_label", "method", "welfare_cost"])

    table_paths = []
    selected_method_by_label = {
        analysis["label"]: analysis["method"]
        for analysis in analyses
    }
    desc_tex = _write_pivot_tex(
        os.path.join(output_dir, f"comparative_descriptive_aggregates_{config['comparative_name']}.tex"),
        rows=descriptive_rows,
        row_filter=lambda row: (
            row.get("variable_group") == "aggregate"
            and row.get("method") == selected_method_by_label.get(row.get("analysis_label"))
        ),
        value_column="mean",
        caption="Comparative aggregate ergodic means",
        label="tab:comparative_descriptive_aggregates",
    )
    if desc_tex:
        table_paths.append(desc_tex)

    stochss_tex = _write_pivot_tex(
        os.path.join(output_dir, f"comparative_stochastic_ss_aggregates_{config['comparative_name']}.tex"),
        rows=stochss_rows,
        row_filter=lambda row: (
            row.get("variable_group") == "aggregate"
            and row.get("method") == selected_method_by_label.get(row.get("analysis_label"))
        ),
        value_column="value_percent",
        caption="Comparative aggregate stochastic steady state",
        label="tab:comparative_stochastic_ss_aggregates",
    )
    if stochss_tex:
        table_paths.append(stochss_tex)

    welfare_tex = _write_pivot_tex(
        os.path.join(output_dir, f"comparative_welfare_{config['comparative_name']}.tex"),
        rows=[
            {**row, "variable": row.get("method", ""), "value": row.get("welfare_cost", "")}
            for row in filtered_welfare_rows
        ],
        row_filter=lambda row: True,
        value_column="value",
        caption="Comparative welfare costs",
        label="tab:comparative_welfare",
    )
    if welfare_tex:
        table_paths.append(welfare_tex)

    figure_paths = _plot_aggregate_comparisons(
        analyses,
        output_dir,
        config.get("aggregate_variables", []),
        int(config.get("max_plot_points", 2000)),
    )
    _write_latex_wrapper(output_dir, config["comparative_name"], table_paths, figure_paths)

    print("\nComparative analysis complete.")
    return {
        "analyses": analyses,
        "descriptive_rows": descriptive_rows,
        "stochastic_ss_rows": stochss_rows,
        "welfare_rows": filtered_welfare_rows,
        "figures": figure_paths,
    }


if __name__ == "__main__":
    main()
