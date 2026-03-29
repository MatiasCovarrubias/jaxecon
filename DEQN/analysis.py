#!/usr/bin/env python3
"""
Analysis script for DEQN trained models.

Usage:
    LOCAL:
        # Method 1: Run as module (from repository root):
        python -m DEQN.analysis

        # Method 2: Run directly as script (from repository root):
        python DEQN/analysis.py

        Both methods require you to be in the repository root directory.

    COLAB:
        Simply run all cells in order. The script will automatically detect the Colab
        environment, install dependencies, clone the repository, and mount Google Drive.
"""

import os
import sys

# ============================================================================
# ENVIRONMENT DETECTION AND SETUP
# ============================================================================

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    print("Installing JAX with CUDA support...")
    import subprocess

    subprocess.run(["pip", "install", "--upgrade", "jax[cuda12]"], check=True)

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    sys.path.insert(0, "/content/jaxecon")

    print("Mounting Google Drive...")
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")

    base_dir = "/content/drive/MyDrive/Jaxecontemp"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import importlib  # noqa: E402
import json  # noqa: E402
from typing import cast  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402
from jax import random  # noqa: E402

from DEQN.analysis.GIR import create_GIR_fn  # noqa: E402
from DEQN.analysis.model_hooks import (  # noqa: E402
    apply_model_config_defaults,
    compute_analysis_variables,
    get_shock_dimension,
    get_states_to_shock,
    load_model_analysis_hooks,
    run_model_postprocess,
)
from DEQN.analysis.simul_analysis import (  # noqa: E402
    compute_analysis_dataset_with_context,
    create_episode_simulation_fn_verbose,
    create_shock_path_simulation_fn,
    simulation_analysis,
    simulation_analysis_with_shocks,
)
from DEQN.analysis.stochastic_ss import (  # noqa: E402
    create_stochss_fn,
    create_stochss_loss_fn,
)
from DEQN.analysis.tables import (  # noqa: E402
    create_calibration_table,
    create_descriptive_stats_table,
    create_stochastic_ss_aggregates_table,
    create_welfare_table,
)
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.training.checkpoints import (  # noqa: E402
    load_experiment_data,
    load_trained_model_orbax,
)

jax_config.update("jax_debug_nans", True)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary
config = {
    # Key configuration - Edit these first
    "model_dir": "RbcProdNet_April2026",
    "analysis_name": "GO_shocks_newWDS_v2",
    # MATLAB data files (relative to model_dir)
    # Set to None to use defaults: "ModelData.mat", "ModelData_IRs.mat", "ModelData_simulation.mat"
    "model_data_file": "ModelData_newwds_v2.mat",
    "model_data_irs_file": "ModelData_IRs_newwds_v2.mat",
    "model_data_simulation_file": "ModelData_simulation_newwds_v2.mat",  # Set to None to skip MATLAB simulation comparison
    # Aggregation convention
    # False (default): use aggregate endogenous policy variables directly from the model / Dynare objects.
    # True: re-aggregate using fixed ergodic-mean prices computed from the nonlinear ergodic simulation.
    "ergodic_price_aggregation": False,
    # Experiments to analyze
    "experiments_to_analyze": {
        "benchmark": "GO_shocks_newWDS_v2",
    },
    # Simulation configuration
    "init_range": 6,
    "periods_per_epis": 64000,
    "burn_in_periods": 3200,
    "simul_vol_scale": 1,
    "simul_seed": 0,
    "n_simul_seeds": 16,
    # Welfare configuration
    "welfare_n_trajects": 16000,
    "welfare_traject_length": 200,
    "welfare_seed": 0,
    # Stochastic steady state configuration
    "n_draws": 2000,
    "time_to_converge": 500,
    "seed": 0,
    # GIR configuration
    "gir_n_draws": 1000,
    "gir_trajectory_length": 100,
    "shock_size": 0.2,
    "gir_seed": 42,
    # IR selection:
    # - False: stochastic-steady-state impulse response
    # - True: generalized impulse response averaged over ergodic draws
    "use_gir": True,
    # MATLAB benchmark overlays used in IR figures.
    # Override with any subset/order of ["PerfectForesight", "FirstOrder", "SecondOrder"].
    "ir_benchmark_methods": ["PerfectForesight", "FirstOrder"],
    # Combined IR analysis configuration
    # Sectors to analyze: specify sector indices (0-based).
    # GIRs shock the TFP/productivity state (state index = n_sectors + sector_idx).
    # For example, sector 0 TFP is at state index 37 (for n_sectors=37).
    "ir_sectors_to_plot": [0],
    "sectoral_ir_variables_to_plot": [
        "Cj",
        "Pj",
        "Ioutj",
        "Moutj",
        "Lj",
        "Ij",
        "Mj",
        "Yj",
        "Qj",
        "Kj",
        "Cj_client",
        "Pj_client",
        "Ioutj_client",
        "Moutj_client",
        "Lj_client",
        "Ij_client",
        "Mj_client",
        "Yj_client",
        "Qj_client",
        "Pmj_client",
        "gammaij_client",
    ],
    "ir_max_periods": 20,
    # Shock sizes are discovered from the MATLAB IR objects.
    # Aggregate tables use all supported aggregates and all available simulations by default.
    # JAX configuration
    "double_precision": True,
}

# ============================================================================
# DYNAMIC IMPORTS (based on model_dir from config)
# ============================================================================

# Import Model class from the specified model directory
model_module = importlib.import_module(f"DEQN.econ_models.{config['model_dir']}.model")
Model = model_module.Model
analysis_hooks = load_model_analysis_hooks(config["model_dir"])
config = apply_model_config_defaults(config, analysis_hooks)

# Import model-specific plots module and registry if available
plots_module_name = f"DEQN.econ_models.{config['model_dir']}.plots"
try:
    plots_module = importlib.import_module(plots_module_name)
except ModuleNotFoundError as exc:
    if exc.name == plots_module_name:
        plots_module = None
    else:
        raise

MODEL_SPECIFIC_PLOTS = getattr(plots_module, "MODEL_SPECIFIC_PLOTS", []) if plots_module is not None else []


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================


DEFAULT_AGGREGATE_LABELS = [
    "Agg. Consumption",
    "Agg. Investment",
    "Agg. GDP",
    "Agg. Capital",
    "Agg. Labor",
    "Intratemporal Utility",
]

DEFAULT_IR_BENCHMARK_METHODS = ["PerfectForesight", "FirstOrder"]


def _write_analysis_config(config_dict, analysis_dir):
    config_path = os.path.join(analysis_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def _analysis_named_path(directory, stem, analysis_name, extension):
    suffix = f"_{analysis_name}" if analysis_name else ""
    return os.path.join(directory, f"{stem}{suffix}{extension}")


def _escape_latex(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _make_safe_plot_label(label):
    return label.replace(" ", "_").replace(".", "").replace("/", "_")


def _latex_relative_path(path, base_dir):
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def _tex_fragment_has_table_env(tex_path):
    if not os.path.exists(tex_path):
        return False
    with open(tex_path) as tex_file:
        return r"\begin{table}" in tex_file.read()


def _figure_note_path(figure_path):
    return os.path.splitext(figure_path)[0] + "_note.tex"


def _join_labels(labels):
    labels = [label for label in labels if label]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _build_simple_figure_spec(path, caption, note_text=None, note_path=None):
    return {
        "path": path,
        "caption": caption,
        "note_text": note_text,
        "note_path": note_path,
    }


def _caption_label(label):
    return label if str(label).isupper() else str(label).lower()


def _format_percent_list(values):
    def _format_percent_value(value):
        rounded = round(float(value), 8)
        if float(rounded).is_integer():
            return str(int(round(rounded)))
        return f"{rounded:.8f}".rstrip("0").rstrip(".")

    values = [_format_percent_value(value) for value in values if value is not None]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _resolve_ir_benchmark_methods(config_dict):
    configured_methods = config_dict.get("ir_benchmark_methods")
    if configured_methods is None:
        legacy_method = config_dict.get("ir_benchmark_method")
        configured_methods = [legacy_method] if legacy_method else list(DEFAULT_IR_BENCHMARK_METHODS)
    elif isinstance(configured_methods, str):
        configured_methods = [configured_methods]

    resolved_methods = []
    for method in configured_methods:
        if method and method not in resolved_methods:
            resolved_methods.append(method)
    return resolved_methods or list(DEFAULT_IR_BENCHMARK_METHODS)


def _describe_ir_benchmark_methods(config_dict):
    benchmark_method_labels = {
        "FirstOrder": "MATLAB first-order benchmark",
        "SecondOrder": "MATLAB second-order benchmark",
        "PerfectForesight": "MATLAB perfect-foresight benchmark",
    }
    labels = [
        benchmark_method_labels.get(method, f"MATLAB {method} benchmark")
        for method in _resolve_ir_benchmark_methods(config_dict)
    ]
    return _join_labels(labels) if labels else "MATLAB benchmarks"


def _existing_subfigures(figure_specs):
    return [figure_spec for figure_spec in figure_specs if os.path.exists(figure_spec["path"])]


def _build_analysis_latex_sections(*, config_dict, analysis_dir, simulation_dir, irs_dir, econ_model):
    analysis_name = config_dict.get("analysis_name") or "analysis"
    sections = []

    def add_table_section(title, tex_paths):
        existing_paths = [path for path in tex_paths if os.path.exists(path)]
        if existing_paths:
            sections.append({"title": title, "tables": existing_paths, "figures": []})

    def add_figure_section(title, figure_paths):
        existing_figures = []
        for figure in figure_paths:
            figure_path = figure["path"] if isinstance(figure, dict) else figure
            if os.path.exists(figure_path):
                existing_figures.append(figure)
        if existing_figures:
            sections.append({"title": title, "tables": [], "figures": existing_figures})

    def add_grouped_figure_section(title, figure_groups):
        existing_groups = []
        for figure_group in figure_groups:
            subfigures = _existing_subfigures(figure_group.get("subfigures", []))
            if subfigures:
                existing_group = dict(figure_group)
                existing_group["subfigures"] = subfigures
                caption_builder = existing_group.get("caption_builder")
                if caption_builder is not None:
                    existing_group["caption"] = caption_builder(subfigures)
                note_builder = existing_group.get("note_builder")
                if note_builder is not None:
                    existing_group["note_text"] = note_builder(subfigures)
                existing_groups.append(existing_group)
        if existing_groups:
            sections.append({"title": title, "tables": [], "figures": existing_groups})

    add_table_section(
        "1. Model vs. Data Moments",
        [os.path.join(analysis_dir, f"calibration_table_{analysis_name}.tex")],
    )

    aggregate_ir_groups = []
    aggregate_variable_captions = {
        "Agg. Consumption": "Consumption",
        "Agg. Investment": "Investment",
        "Agg. GDP": "GDP",
        "Agg. Labor": "Labor",
        "Agg. Capital": "Capital",
        "Intratemporal Utility": "Intratemporal Utility",
    }
    aggregate_variable_note_labels = {
        "Agg. Consumption": "consumption",
        "Agg. Investment": "investment",
        "Agg. GDP": "GDP",
        "Agg. Labor": "labor",
        "Agg. Capital": "capital",
        "Intratemporal Utility": "intratemporal utility",
    }
    ir_shock_sizes = list(config_dict.get("ir_shock_sizes", []))
    aggregate_benchmark_labels = _describe_ir_benchmark_methods(config_dict)
    aggregate_ir_variables = list(getattr(analysis_hooks, "DEFAULT_AGGREGATE_IR_LABELS", DEFAULT_AGGREGATE_LABELS))
    for sector_idx in config_dict.get("ir_sectors_to_plot", []):
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        safe_sector = _make_safe_plot_label(sector_label)
        aggregate_specs = []
        for variable_name in aggregate_ir_variables:
            safe_variable = _make_safe_plot_label(variable_name)
            figure_spec = {
                "path": _analysis_named_path(irs_dir, f"IR_{safe_variable}_{safe_sector}", analysis_name, ".png"),
                "caption": aggregate_variable_captions.get(variable_name, variable_name),
                "note_label": aggregate_variable_note_labels.get(variable_name, variable_name),
            }
            aggregate_specs.append(figure_spec)

        if aggregate_specs:
            aggregate_ir_groups.append(
                {
                    "caption_builder": lambda subfigures, sector_label=sector_label: (
                        f"Aggregate {_join_labels([_caption_label(subfigure['caption']) for subfigure in subfigures])} "
                        f"responses to a TFP shock in {sector_label}."
                    ),
                    "note_builder": lambda subfigures, sector_label=sector_label, ir_shock_sizes=tuple(
                        ir_shock_sizes
                    ), aggregate_benchmark_labels=aggregate_benchmark_labels: (
                        f"Each row corresponds to a {_format_percent_list(ir_shock_sizes)} percent TFP shock in "
                        f"{sector_label}; the left column shows negative shocks and the right column positive shocks. "
                        f"The panels plot the responses of aggregate "
                        f"{_join_labels([subfigure.get('note_label', _caption_label(subfigure['caption'])) for subfigure in subfigures])} "
                        f"MATLAB benchmark IRs are anchored at the deterministic steady state, "
                        f"while global-solution IRs start from and return to the stochastic steady state. "
                        f"The horizontal axis reports periods after impact. "
                        f"The vertical axis reports impulse responses in percent. "
                        f"Solid lines report the DEQN stochastic-steady-state impulse response and dashed lines report the "
                        f"{aggregate_benchmark_labels}; all benchmark overlays are aggregated with fixed ergodic-price weights."
                    ),
                    "subfigures": aggregate_specs,
                }
            )
    add_grouped_figure_section("2. Aggregate Impulse Responses", aggregate_ir_groups)

    add_figure_section(
        "3. Sectoral Variables in Stochastic Steady State",
        [
            _build_simple_figure_spec(
                _analysis_named_path(simulation_dir, f"sectoral_{variable_name}_stochss", analysis_name, ".png"),
                f"Sectoral {variable_caption} at the stochastic steady state.",
            )
            for variable_name, variable_caption in [
                ("k", "capital"),
                ("l", "labor"),
                ("y", "value added"),
                ("m", "intermediates"),
                ("q", "gross output"),
            ]
        ],
    )

    add_table_section(
        "4. Aggregate Stochastic Steady State",
        [
            os.path.join(analysis_dir, f"stochastic_ss_aggregates_{analysis_name}.tex"),
        ],
    )

    add_table_section(
        "5. Descriptive Statistics",
        [
            os.path.join(simulation_dir, f"descriptive_stats_{analysis_name}.tex"),
        ],
    )

    add_table_section(
        "6. Welfare Cost of Business Cycles",
        [os.path.join(analysis_dir, f"welfare_{analysis_name}.tex")],
    )

    sectoral_ir_groups = []
    largest_sectoral_shock = max(config_dict.get("ir_shock_sizes", [0])) if config_dict.get("ir_shock_sizes") else None
    sectoral_benchmark_labels = _describe_ir_benchmark_methods(config_dict)
    sectoral_group_specs = [
        (
            "Shocked Sector Inputs",
            [("Lj", "Labor"), ("Ij", "Investment"), ("Mj", "Intermediates"), ("Yj", "Value Added"), ("Kj", "Capital")],
        ),
        (
            "Shocked Sector Outputs",
            [
                ("Cj", "Consumption"),
                ("Pj", "Price"),
                ("Moutj", "Intermediate Sales"),
                ("Ioutj", "Investment Sales"),
                ("Qj", "Gross Output"),
            ],
        ),
        (
            "Client Sector Inputs",
            [
                ("Lj_client", "Labor"),
                ("Ij_client", "Investment"),
                ("Mj_client", "Intermediates"),
                ("Yj_client", "Value Added"),
                ("Pmj_client", "Intermediate Price"),
                ("gammaij_client", "Expenditure Share"),
            ],
        ),
        (
            "Client Sector Outputs",
            [
                ("Cj_client", "Consumption"),
                ("Pj_client", "Price"),
                ("Moutj_client", "Intermediate Sales"),
                ("Ioutj_client", "Investment Sales"),
                ("Qj_client", "Gross Output"),
            ],
        ),
    ]
    for sector_idx in config_dict.get("ir_sectors_to_plot", []):
        sector_label = (
            econ_model.labels[sector_idx] if sector_idx < len(econ_model.labels) else f"Sector {sector_idx + 1}"
        )
        safe_sector = _make_safe_plot_label(sector_label)
        configured_sectoral_variables = set(config_dict.get("sectoral_ir_variables_to_plot", []))
        for group_title, variable_specs in sectoral_group_specs:
            subfigures = []
            for variable_name, variable_caption in variable_specs:
                if variable_name not in configured_sectoral_variables:
                    continue
                safe_variable = _make_safe_plot_label(variable_name)
                subfigures.append(
                    {
                        "path": _analysis_named_path(
                            irs_dir,
                            f"IR_{safe_variable}_{safe_sector}",
                            analysis_name,
                            ".png",
                        ),
                        "caption": variable_caption,
                    }
                )

            if subfigures:
                shock_text = (
                    f"The panels plot responses to a negative {largest_sectoral_shock} percent TFP shock in {sector_label}."
                    if largest_sectoral_shock
                    else f"The panels plot responses to the sectoral TFP shock used in the analysis for {sector_label}."
                )
                sectoral_ir_groups.append(
                    {
                        "caption": f"{group_title} for {sector_label}.",
                        "note_text": (
                            f"Follows the MATLAB grouping. {shock_text} "
                            "MATLAB benchmark IRs are anchored at the deterministic steady state, "
                            "while global-solution IRs start from and return to the stochastic steady state. "
                            "The horizontal axis reports periods after impact. "
                            "The vertical axis reports impulse responses in percent. "
                            f"Dashed lines report the {sectoral_benchmark_labels}."
                        ),
                        "subfigures": subfigures,
                    }
                )
    add_grouped_figure_section("7. Sectoral Impulse Responses", sectoral_ir_groups)

    add_figure_section(
        "8. Ergodic Mean Sectoral Variables",
        [
            _build_simple_figure_spec(
                _analysis_named_path(simulation_dir, f"sectoral_{variable_name}_ergodic", analysis_name, ".png"),
                f"Ergodic mean sectoral {variable_caption}.",
            )
            for variable_name, variable_caption in [
                ("k", "capital"),
                ("l", "labor"),
                ("y", "value added"),
                ("m", "intermediates"),
                ("q", "gross output"),
            ]
        ],
    )

    return sections


def _write_analysis_results_latex(*, config_dict, analysis_dir, simulation_dir, irs_dir, econ_model):
    analysis_name = config_dict.get("analysis_name") or "analysis"
    sections = _build_analysis_latex_sections(
        config_dict=config_dict,
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        irs_dir=irs_dir,
        econ_model=econ_model,
    )
    if not sections:
        print("  ⚠ Combined LaTeX file skipped: no rendered tables or figures were found.")
        return None

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{textcomp}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{graphicx}",
        r"\usepackage{subcaption}",
        r"\usepackage{float}",
        r"\floatplacement{table}{H}",
        r"\floatplacement{figure}{H}",
        r"\newlength{\FigHfirst}",
        r"\newlength{\FigHsingle}",
        r"\newlength{\FigHdouble}",
        r"\newlength{\FigHpanel}",
        r"\setlength{\FigHfirst}{0.70\textheight}",
        r"\setlength{\FigHsingle}{0.58\textheight}",
        r"\setlength{\FigHdouble}{0.40\textheight}",
        r"\setlength{\FigHpanel}{0.23\textheight}",
        r"\captionsetup[subfigure]{justification=centering}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.75em}",
        r"\begin{document}",
    ]

    rendered_figure_count = 0

    def _single_figure_include_options(*, is_first_figure):
        height_name = r"\FigHfirst" if is_first_figure else r"\FigHsingle"
        return rf"width=0.96\textwidth,height={height_name},keepaspectratio"

    def _subfigure_layout(*, subfigure_count, is_first_figure):
        if subfigure_count >= 5:
            return "0.43\\textwidth", r"\FigHpanel", r"\hspace{0.04\textwidth}"
        if subfigure_count > 1:
            return "0.48\\textwidth", r"\FigHdouble", r"\hfill"
        if is_first_figure:
            return r"\textwidth", r"\FigHfirst", None
        return "0.88\\textwidth", r"\FigHsingle", None

    for section in sections:
        for tex_path in section["tables"]:
            relative_path = _latex_relative_path(tex_path, analysis_dir)
            if _tex_fragment_has_table_env(tex_path):
                lines.append(rf"\input{{{relative_path}}}")
            else:
                lines.extend(
                    [
                        r"\begin{table}[H]",
                        r"\centering",
                        rf"\input{{{relative_path}}}",
                        r"\end{table}",
                    ]
                )
            lines.append(r"\clearpage")

        for figure_path in section["figures"]:
            if isinstance(figure_path, str) or "subfigures" not in figure_path:
                single_figure = (
                    _build_simple_figure_spec(figure_path, "") if isinstance(figure_path, str) else figure_path
                )
                is_first_figure = rendered_figure_count == 0
                relative_path = _latex_relative_path(single_figure["path"], analysis_dir)
                note_path = single_figure.get("note_path") or _figure_note_path(single_figure["path"])
                lines.extend(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[{_single_figure_include_options(is_first_figure=is_first_figure)}]{{{relative_path}}}",
                    ]
                )
                if single_figure.get("caption"):
                    lines.append(rf"\caption{{{_escape_latex(single_figure['caption'])}}}")
                if os.path.exists(note_path):
                    lines.extend([r"\par\smallskip", rf"\input{{{_latex_relative_path(note_path, analysis_dir)}}}"])
                elif single_figure.get("note_text"):
                    lines.extend(
                        [
                            r"\par\smallskip",
                            r"\begin{minipage}{0.92\textwidth}",
                            r"\footnotesize",
                            rf"\textit{{Notes:}} {_escape_latex(single_figure['note_text'])}",
                            r"\end{minipage}",
                        ]
                    )
                lines.extend([r"\end{figure}", r"\clearpage"])
                rendered_figure_count += 1
                continue

            subfigures = figure_path.get("subfigures", [])
            if not subfigures:
                continue

            is_first_figure = rendered_figure_count == 0
            lines.extend([r"\begin{figure}[H]", r"\centering"])
            width, height_name, column_separator = _subfigure_layout(
                subfigure_count=len(subfigures),
                is_first_figure=is_first_figure,
            )
            for idx, subfigure in enumerate(subfigures):
                lines.extend(
                    [
                        rf"\begin{{subfigure}}[t]{{{width}}}",
                        r"\centering",
                        rf"\includegraphics[width=\linewidth,height={height_name},keepaspectratio]{{{_latex_relative_path(subfigure['path'], analysis_dir)}}}",
                        rf"\caption{{{_escape_latex(subfigure.get('caption', ''))}}}",
                        r"\end{subfigure}",
                    ]
                )
                if len(subfigures) > 1 and idx % 2 == 0 and idx != len(subfigures) - 1:
                    lines.append(column_separator or r"\hfill")
                elif idx != len(subfigures) - 1:
                    lines.append(r"\par\medskip")

            lines.append(rf"\caption{{{_escape_latex(figure_path.get('caption', ''))}}}")
            note_path = figure_path.get("note_path")
            note_text = figure_path.get("note_text")
            if note_path and os.path.exists(note_path):
                lines.extend([r"\par\smallskip", rf"\input{{{_latex_relative_path(note_path, analysis_dir)}}}"])
            elif note_text:
                lines.extend(
                    [
                        r"\par\smallskip",
                        r"\begin{minipage}{0.92\textwidth}",
                        r"\footnotesize",
                        rf"\textit{{Notes:}} {_escape_latex(note_text)}",
                        r"\end{minipage}",
                    ]
                )
            lines.extend([r"\end{figure}", r"\clearpage"])
            rendered_figure_count += 1

    lines.append(r"\end{document}")

    combined_tex_path = os.path.join(analysis_dir, f"figures_tables_{analysis_name}.tex")
    with open(combined_tex_path, "w") as combined_file:
        combined_file.write("\n".join(lines) + "\n")

    table_count = sum(len(section["tables"]) for section in sections)
    figure_count = sum(len(section["figures"]) for section in sections)
    print(f"  Combined LaTeX file saved: {combined_tex_path}")
    print(f"  Included {table_count} tables and {figure_count} figures.")
    return combined_tex_path


def _resolve_data_file(model_dir, configured_name, fallback_names, *, label, required):
    candidate_names = []
    for name in [configured_name, *fallback_names]:
        if name is None or name in candidate_names:
            continue
        candidate_names.append(name)

    for filename in candidate_names:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            if configured_name is not None and filename != configured_name:
                print(f"  Using fallback {label} file: {filename} (configured '{configured_name}' not found)")
            return filename, path

    if required:
        raise FileNotFoundError(f"{label} file not found in {model_dir}. Tried: {candidate_names}")

    if configured_name is not None:
        print(f"  ⚠ {label} file not found. Tried: {candidate_names} (skipping)")
    return None, None


def _create_nonlinear_simulation_runners(
    *,
    econ_model,
    config_dict,
    analysis_hooks,
    matlab_common_shock_schedule,
):
    ergodic_simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config_dict))
    print("  Nonlinear simulation mode: long ergodic simulation")

    def run_ergodic_simulation(train_state):
        simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
            train_state=train_state,
            econ_model=econ_model,
            analysis_config=config_dict,
            simulation_fn=ergodic_simulation_fn,
            analysis_hooks=analysis_hooks,
        )
        return {
            "simul_obs": simul_obs,
            "simul_policies": simul_policies,
            "simul_analysis_variables": simul_analysis_variables,
            "analysis_context": analysis_context,
            "simul_obs_full": simul_obs,
            "simul_policies_full": simul_policies,
            "simulation_kind": "ergodic",
        }

    common_shock_runner = None
    if matlab_common_shock_schedule is not None:
        shock_path_simulation_fn = jax.jit(create_shock_path_simulation_fn(econ_model))
        print(
            "  Nonlinear simulation mode: shared MATLAB shock path kept as auxiliary run "
            f"({matlab_common_shock_schedule['reference_method']})"
        )

        def run_common_shock_simulation(train_state):
            (
                simul_obs,
                simul_policies,
                simul_analysis_variables,
                analysis_context,
                simul_obs_full,
                simul_policies_full,
            ) = simulation_analysis_with_shocks(
                train_state=train_state,
                econ_model=econ_model,
                shock_path=matlab_common_shock_schedule["full_shocks"],
                simulation_fn=shock_path_simulation_fn,
                active_start=matlab_common_shock_schedule["active_start"],
                active_end=matlab_common_shock_schedule["active_end"],
                analysis_config=config_dict,
                analysis_hooks=analysis_hooks,
                label=(f"Common-shock nonlinear simulation ({matlab_common_shock_schedule['reference_method']})"),
            )
            return {
                "simul_obs": simul_obs,
                "simul_policies": simul_policies,
                "simul_analysis_variables": simul_analysis_variables,
                "analysis_context": analysis_context,
                "simul_obs_full": simul_obs_full,
                "simul_policies_full": simul_policies_full,
                "simulation_kind": "common_shock",
            }

        common_shock_runner = run_common_shock_simulation
    else:
        print("  No MATLAB common-shock schedule found; only long ergodic simulation will be used.")

    return {
        "ergodic": run_ergodic_simulation,
        "common_shock": common_shock_runner,
    }


def _common_shock_label(experiment_label: str) -> str:
    return f"{experiment_label} (common shocks)"


def _build_output_display_label_map(config_dict):
    experiment_labels = list((config_dict.get("experiments_to_analyze") or {}).keys())
    if len(experiment_labels) != 1:
        return {}

    experiment_label = experiment_labels[0]
    return {
        experiment_label: "Global Solution",
        _common_shock_label(experiment_label): "Global Solution (Common Shocks)",
    }


def _apply_display_labels_to_mapping(values_by_label, label_map):
    if not label_map:
        return dict(values_by_label)

    relabeled = {}
    for label, value in values_by_label.items():
        relabeled[label_map.get(label, label)] = value
    return relabeled


def _apply_display_labels_to_postprocess_context(postprocess_context, label_map):
    if not postprocess_context or not label_map:
        return postprocess_context

    relabeled_context = dict(postprocess_context)

    ergodic_labels = postprocess_context.get("ergodic_experiment_labels")
    if ergodic_labels is not None:
        relabeled_context["ergodic_experiment_labels"] = [label_map.get(label, label) for label in ergodic_labels]

    reference_label = postprocess_context.get("reference_experiment_label")
    if reference_label is not None:
        relabeled_context["reference_experiment_label"] = label_map.get(reference_label, reference_label)

    return relabeled_context


def _apply_display_labels_to_sequence(labels, label_map):
    if labels is None:
        return None
    return [label_map.get(label, label) for label in labels]


def _compute_welfare_cost_from_sample(*, econ_model, welfare_fn, welfare_ss, policies_logdev, config_dict):
    simul_utilities = jax.vmap(econ_model.utility_from_policies)(policies_logdev)
    welfare = welfare_fn(simul_utilities, welfare_ss, random.PRNGKey(config_dict["welfare_seed"]))
    return -econ_model.consumption_equivalent(welfare) * 100


def _compute_stochastic_ss_from_sample(
    *,
    sample_label,
    simul_obs,
    train_state,
    stoch_ss_fn,
    stoch_ss_loss_fn,
    analysis_context,
    econ_model,
    analysis_hooks,
    config_dict,
    required,
):
    try:
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
        if stoch_ss_obs_std.max() > 0.001:
            raise ValueError("Stochastic steady state standard deviation too large")

        stoch_ss_analysis_variables = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=stoch_ss_obs,
            policy_logdev=stoch_ss_policy,
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )

        loss_results = stoch_ss_loss_fn(
            stoch_ss_obs,
            stoch_ss_policy,
            train_state,
            random.PRNGKey(config_dict["seed"]),
        )
        print(
            f"    {sample_label}: equilibrium accuracy {loss_results['mean_accuracy']:.4f} "
            f"(min: {loss_results['min_accuracy']:.4f})",
            flush=True,
        )
        return {
            "stochastic_ss_state": stoch_ss_obs,
            "stochastic_ss_policy": stoch_ss_policy,
            "stochastic_ss_data": stoch_ss_analysis_variables,
            "stochastic_ss_loss": loss_results,
        }
    except Exception:
        if required:
            raise
        print(f"    Warning: stochastic steady state failed for {sample_label}; skipping this variant.", flush=True)
        return None


def _run_experiment_analysis(
    *,
    experiment_label,
    exp_data,
    save_dir,
    nn_config_base,
    econ_model,
    nonlinear_simulation_runners,
    welfare_fn,
    welfare_ss,
    stoch_ss_fn,
    stoch_ss_loss_fn,
    gir_fn,
    config_dict,
    analysis_hooks,
):
    experiment_config = exp_data["config"]
    experiment_name = exp_data["results"]["exper_name"]

    nn_config = nn_config_base.copy()
    nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

    train_state = load_trained_model_orbax(experiment_name, save_dir, nn_config, econ_model.state_ss)

    ergodic_results = nonlinear_simulation_runners["ergodic"](train_state)
    ergodic_analysis_context = ergodic_results["analysis_context"]

    welfare_costs = {}
    raw_simulation_data = {}
    analysis_variables = {}
    stochastic_ss_states = {}
    stochastic_ss_policies = {}
    stochastic_ss_data = {}
    stochastic_ss_loss = {}
    method_labels = []

    def store_variant(label, sample_results, *, analysis_context_for_reporting, required_stochss):
        welfare_cost_ce = _compute_welfare_cost_from_sample(
            econ_model=econ_model,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            policies_logdev=sample_results["simul_policies"],
            config_dict=config_dict,
        )
        welfare_costs[label] = welfare_cost_ce
        print(f"    {label}: welfare cost (CE) {welfare_cost_ce:.4f}%")

        raw_entry = dict(sample_results)
        raw_entry["analysis_context"] = analysis_context_for_reporting
        raw_simulation_data[label] = raw_entry
        analysis_variables[label] = sample_results["simul_analysis_variables"]
        method_labels.append(label)

        stochss_results = _compute_stochastic_ss_from_sample(
            sample_label=label,
            simul_obs=sample_results["simul_obs"],
            train_state=train_state,
            stoch_ss_fn=stoch_ss_fn,
            stoch_ss_loss_fn=stoch_ss_loss_fn,
            analysis_context=analysis_context_for_reporting,
            econ_model=econ_model,
            analysis_hooks=analysis_hooks,
            config_dict=config_dict,
            required=required_stochss,
        )
        if stochss_results is not None:
            stochastic_ss_states[label] = stochss_results["stochastic_ss_state"]
            stochastic_ss_policies[label] = stochss_results["stochastic_ss_policy"]
            stochastic_ss_data[label] = stochss_results["stochastic_ss_data"]
            stochastic_ss_loss[label] = stochss_results["stochastic_ss_loss"]
        return welfare_cost_ce

    store_variant(
        experiment_label,
        ergodic_results,
        analysis_context_for_reporting=ergodic_analysis_context,
        required_stochss=True,
    )

    common_shock_runner = nonlinear_simulation_runners.get("common_shock")
    if common_shock_runner is not None:
        common_shock_results = common_shock_runner(train_state)
        common_shock_analysis_variables, _ = compute_analysis_dataset_with_context(
            econ_model=econ_model,
            simul_obs=common_shock_results["simul_obs"],
            simul_policies=common_shock_results["simul_policies"],
            analysis_config=config_dict,
            analysis_context=ergodic_analysis_context,
            analysis_hooks=analysis_hooks,
        )
        common_shock_results["simul_analysis_variables"] = common_shock_analysis_variables
        store_variant(
            _common_shock_label(experiment_label),
            common_shock_results,
            analysis_context_for_reporting=ergodic_analysis_context,
            required_stochss=False,
        )

    gir_results = gir_fn(
        ergodic_results["simul_obs"],
        train_state,
        ergodic_results["simul_policies"],
        stochastic_ss_states[experiment_label],
    )

    return {
        "raw_simulation_data": raw_simulation_data,
        "analysis_variables": analysis_variables,
        "welfare_costs": welfare_costs,
        "stochastic_ss_states": stochastic_ss_states,
        "stochastic_ss_policies": stochastic_ss_policies,
        "stochastic_ss_data": stochastic_ss_data,
        "stochastic_ss_loss": stochastic_ss_loss,
        "gir_data": gir_results,
        "nonlinear_method_labels": method_labels,
    }


def _normalize_dynare_simulation_orientation(simul_matrix, expected_n_vars, precision):
    arr = jnp.array(simul_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((expected_n_vars, 0), dtype=precision)
        if arr.size == expected_n_vars:
            return arr.reshape(expected_n_vars, 1)
        raise ValueError(
            "Unexpected 1D Dynare simulation vector with "
            f"length={arr.size}; expected length {expected_n_vars} for a single-period slice."
        )
    if arr.ndim != 2:
        raise ValueError(f"Unexpected Dynare simulation ndim={arr.ndim}; expected 1 or 2.")
    if arr.shape[0] == expected_n_vars:
        return arr
    if arr.shape[1] == expected_n_vars:
        return arr.T
    raise ValueError(f"Unexpected Dynare simulation shape {arr.shape}; expected one axis = {expected_n_vars}.")


def _extract_dynare_simulation_artifact(simul, method_names, expected_n_vars, precision):
    """Load active/full simulation paths for one Dynare method."""
    for method_name in method_names:
        method_block = simul.get(method_name)
        if not isinstance(method_block, dict):
            continue

        full_simul = None
        active_simul = None

        full_simul_raw = method_block.get("full_simul")
        if full_simul_raw is not None:
            full_simul = _normalize_dynare_simulation_orientation(full_simul_raw, expected_n_vars, precision)

        burnin_simul = method_block.get("burnin_simul")
        shocks_simul = method_block.get("shocks_simul")
        burnout_simul = method_block.get("burnout_simul")

        if shocks_simul is not None:
            active_simul = _normalize_dynare_simulation_orientation(shocks_simul, expected_n_vars, precision)

        if full_simul is None:
            windows = []
            for window in (burnin_simul, shocks_simul, burnout_simul):
                if window is None:
                    continue
                window_arr = _normalize_dynare_simulation_orientation(window, expected_n_vars, precision)
                if window_arr.shape[1] > 0:
                    windows.append(window_arr)
            if windows:
                full_simul = jnp.concatenate(windows, axis=1)

        if active_simul is None and full_simul is not None:
            burn_in = int(method_block.get("burn_in", 0))
            t_active = method_block.get("T_active")
            if t_active is not None:
                t_active = int(t_active)
                active_simul = full_simul[:, burn_in : burn_in + t_active]
            else:
                active_simul = full_simul

        if active_simul is not None or full_simul is not None:
            return {
                "active_simul": active_simul if active_simul is not None else full_simul,
                "full_simul": full_simul if full_simul is not None else active_simul,
            }

    return {"active_simul": None, "full_simul": None}


def _normalize_shock_matrix(shocks_matrix, shock_dimension, precision):
    arr = jnp.array(shocks_matrix, dtype=precision)
    if arr.ndim == 1:
        if arr.size == 0:
            return jnp.zeros((0, shock_dimension), dtype=precision)
        if arr.size == shock_dimension:
            return arr.reshape(1, shock_dimension)
        raise ValueError(
            "Unexpected 1D shock vector with "
            f"length={arr.size}; expected length {shock_dimension} for a single-period shock path."
        )
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D shock matrix, got shape {arr.shape}")
    if arr.shape[1] == shock_dimension:
        return arr
    if arr.shape[0] == shock_dimension:
        return arr.T
    raise ValueError(f"Unexpected shock matrix shape {arr.shape} for shock_dimension={shock_dimension}")


def _extract_matlab_common_shock_schedule(simul, shock_dimension, precision):
    shocks_block = simul.get("Shocks")
    if not isinstance(shocks_block, dict) or "data" not in shocks_block:
        return None

    active_shocks_full = _normalize_shock_matrix(shocks_block["data"], shock_dimension, precision)
    usage = shocks_block.get("usage", {})

    candidate_methods = [
        ("FirstOrder", ["FirstOrder", "Loglin", "LogLinear"]),
        ("SecondOrder", ["SecondOrder"]),
        ("PerfectForesight", ["PerfectForesight", "Determ"]),
        ("MITShocks", ["MITShocks", "MITShock"]),
    ]

    selected_method = None
    method_block = None
    usage_block = None
    for canonical_name, aliases in candidate_methods:
        for alias in aliases:
            block = simul.get(alias)
            if not isinstance(block, dict):
                continue
            usage_candidate = usage.get(alias) or usage.get(canonical_name)
            if usage_candidate is not None or "T_active" in block or "burn_in" in block:
                selected_method = canonical_name
                method_block = block
                usage_block = usage_candidate
                break
        if selected_method is not None:
            break

    if method_block is None:
        return None

    active_shocks = active_shocks_full
    if isinstance(usage_block, dict) and "start" in usage_block and "end" in usage_block:
        start_idx = max(int(usage_block["start"]) - 1, 0)
        end_idx = min(int(usage_block["end"]), active_shocks_full.shape[0])
        active_shocks = active_shocks_full[start_idx:end_idx]

    burn_in = int(method_block.get("burn_in", 0))
    burn_out = int(method_block.get("burn_out", 0))
    zero_burnin = jnp.zeros((burn_in, shock_dimension), dtype=precision)
    zero_burnout = jnp.zeros((burn_out, shock_dimension), dtype=precision)
    full_shocks = jnp.concatenate([zero_burnin, active_shocks, zero_burnout], axis=0)

    return {
        "reference_method": selected_method,
        "active_shocks": active_shocks,
        "full_shocks": full_shocks,
        "burn_in": burn_in,
        "burn_out": burn_out,
        "active_start": burn_in,
        "active_end": burn_in + active_shocks.shape[0],
    }


def _normalize_dynare_full_simul(simul_data, state_ss_vec, policies_ss_vec):
    """Return simulation in log deviations with shape (n_vars, T)."""
    expected_n_vars = state_ss_vec.shape[0] + policies_ss_vec.shape[0]
    if simul_data.shape[0] == expected_n_vars:
        simul_matrix = simul_data
    elif simul_data.shape[1] == expected_n_vars:
        simul_matrix = simul_data.T
    else:
        raise ValueError(
            f"Unexpected Dynare simulation shape {simul_data.shape}; expected one axis = {expected_n_vars}."
        )

    ss_full = jnp.concatenate([state_ss_vec, policies_ss_vec])
    dist_to_zero = jnp.mean(jnp.abs(simul_matrix[:, 0]))
    dist_to_ss = jnp.mean(jnp.abs(simul_matrix[:, 0] - ss_full))
    if dist_to_ss < dist_to_zero:
        simul_matrix = simul_matrix - ss_full[:, None]
    return simul_matrix


def _welfare_cost_from_dynare_simul(
    simul_data,
    method_name,
    state_ss,
    policies_ss,
    econ_model,
    welfare_fn,
    welfare_ss,
    config_dict,
):
    """Compute consumption-equivalent welfare cost from the canonical active Dynare sample."""
    if simul_data is None:
        return None
    simul_matrix = _normalize_dynare_full_simul(simul_data, state_ss, policies_ss)
    n_state_vars = state_ss.shape[0]
    policies_logdev = simul_matrix[n_state_vars:, :].T

    if policies_logdev.shape[0] == 0:
        print(f"  ⚠ Skipping welfare for {method_name}: active sample is empty.")
        return None

    method_seed = sum(ord(c) for c in method_name)
    welfare = welfare_fn(
        jax.vmap(econ_model.utility_from_policies)(policies_logdev),
        welfare_ss,
        random.PRNGKey(config_dict["welfare_seed"] + method_seed),
    )
    Vc = econ_model.consumption_equivalent(welfare)
    return -Vc * 100


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    model_dir = os.path.join(base_dir, config["model_dir"])
    save_dir = os.path.join(model_dir, "experiments/")

    analysis_dir = os.path.join(model_dir, "analysis", config["analysis_name"])
    simulation_dir = os.path.join(analysis_dir, "simulation")
    irs_dir = os.path.join(analysis_dir, "IRs")

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(simulation_dir, exist_ok=True)
    os.makedirs(irs_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD MODEL DATA
    # ═══════════════════════════════════════════════════════════════════════════
    model_data_file, model_path = _resolve_data_file(
        model_dir,
        config.get("model_data_file"),
        ["ModelData.mat", "model_data.mat"],
        label="ModelData",
        required=True,
    )
    model_data_irs_file, irs_path = _resolve_data_file(
        model_dir,
        config.get("model_data_irs_file"),
        ["ModelData_IRs.mat"],
        label="IR benchmark",
        required=False,
    )
    model_data_simulation_file, simul_path = _resolve_data_file(
        model_dir,
        config.get("model_data_simulation_file"),
        ["ModelData_simulation.mat"],
        label="Simulation benchmark",
        required=False,
    )

    config["model_data_file"] = model_data_file
    config["model_data_irs_file"] = model_data_irs_file
    config["model_data_simulation_file"] = model_data_simulation_file
    _write_analysis_config(config, analysis_dir)

    print(f"  Loading ModelData from: {model_data_file}")
    model_data = sio.loadmat(model_path, simplify_cells=True)

    if "ModelData" not in model_data:
        raise ValueError("Expected 'ModelData' key in model file.")

    md = model_data["ModelData"]

    ss = md["SteadyState"]
    n_sectors = ss["parameters"]["parn_sectors"]
    a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
    k_ss = jnp.array(ss["endostates_ss"], dtype=precision)
    state_ss = jnp.concatenate([k_ss, a_ss])
    params = ss["parameters"]
    policies_ss = jnp.array(ss["policies_ss"], dtype=precision)

    stats = md["Statistics"]
    state_sd = jnp.array(stats["states_sd"], dtype=precision)
    policies_sd = jnp.array(stats["policies_sd"], dtype=precision)

    C_matrix = md["Solution"]["StateSpace"]["C"]

    if len(policies_ss) != len(policies_sd):
        n_policies = len(policies_sd)
        policies_ss = policies_ss[:n_policies]

    expected_n_vars = state_ss.shape[0] + policies_ss.shape[0]

    # Load simulation data (optional - for Dynare comparison)
    matlab_common_shock_schedule = None
    matlab_simulation_block = None
    dynare_1st_artifact = {"active_simul": None, "full_simul": None}
    dynare_so_artifact = {"active_simul": None, "full_simul": None}
    dynare_pf_artifact = {"active_simul": None, "full_simul": None}
    dynare_mit_artifact = {"active_simul": None, "full_simul": None}

    if simul_path is not None:
        print(f"  Loading simulation data from: {model_data_simulation_file}")
        simul_data = sio.loadmat(simul_path, simplify_cells=True)
        matlab_simulation_block = simul_data.get("ModelData_simulation", {})

        dynare_1st_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["FirstOrder", "Loglin"], expected_n_vars, precision
        )
        dynare_pf_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["PerfectForesight", "Determ"], expected_n_vars, precision
        )
        dynare_so_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["SecondOrder"], expected_n_vars, precision
        )
        dynare_mit_artifact = _extract_dynare_simulation_artifact(
            matlab_simulation_block, ["MITShocks", "MITShock"], expected_n_vars, precision
        )

    if irs_path:
        print(f"  Found IRs file: {model_data_irs_file}")
    elif config.get("model_data_irs_file"):
        print(f"  ⚠ IRs file not found: {config['model_data_irs_file']} (will try legacy format)")

    # Create economic model
    econ_model = Model(
        parameters=params,
        state_ss=state_ss,
        policies_ss=policies_ss,
        state_sd=state_sd,
        policies_sd=policies_sd,
        double_precision=config["double_precision"],
    )
    shock_dimension = get_shock_dimension(econ_model, analysis_hooks)

    if analysis_hooks is not None and hasattr(analysis_hooks, "discover_ir_shock_sizes"):
        discovered_ir_shock_sizes = analysis_hooks.discover_ir_shock_sizes(
            config=config,
            model_dir=model_dir,
            irs_path=irs_path,
        )
        if discovered_ir_shock_sizes:
            config["ir_shock_sizes"] = list(discovered_ir_shock_sizes)
            print(
                "  Using IR shock sizes discovered from MATLAB objects for DEQN IR computation: "
                f"{config['ir_shock_sizes']}"
            )

    if matlab_simulation_block is not None:
        matlab_common_shock_schedule = _extract_matlab_common_shock_schedule(
            matlab_simulation_block,
            shock_dimension,
            precision,
        )
        if matlab_common_shock_schedule is not None:
            print(
                "  Loaded shared MATLAB shock path "
                f"({matlab_common_shock_schedule['reference_method']}: "
                f"{matlab_common_shock_schedule['burn_in']} burn-in, "
                f"{matlab_common_shock_schedule['active_shocks'].shape[0]} active, "
                f"{matlab_common_shock_schedule['burn_out']} burn-out)"
            )

    # Load experiment data
    experiments_to_analyze = config["experiments_to_analyze"]
    experiments_data = load_experiment_data(
        experiments_to_analyze,
        save_dir,
        expected_model_dir=config["model_dir"],
    )

    nn_config_base = {
        "C": C_matrix,
        "states_sd": state_sd,
        "policies_sd": policies_sd,
        "params_dtype": precision,
    }

    config["states_to_shock"] = get_states_to_shock(
        config=config,
        econ_model=econ_model,
        analysis_hooks=analysis_hooks,
    )

    # Keep welfare baseline available for all methods.
    welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)

    # Create analysis functions
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config))
    stoch_ss_fn = jax.jit(create_stochss_fn(econ_model, config, analysis_hooks=analysis_hooks))
    stoch_ss_loss_fn = create_stochss_loss_fn(econ_model, mc_draws=32)
    gir_fn = jax.jit(create_GIR_fn(econ_model, config, analysis_hooks=analysis_hooks))
    nonlinear_simulation_runners = _create_nonlinear_simulation_runners(
        econ_model=econ_model,
        config_dict=config,
        analysis_hooks=analysis_hooks,
        matlab_common_shock_schedule=matlab_common_shock_schedule,
    )

    # Storage for analysis results
    analysis_variables_data = {}
    raw_simulation_data = {}
    welfare_costs = {}
    stochastic_ss_data = {}
    stochastic_ss_states = {}
    stochastic_ss_policies = {}
    stochastic_ss_loss = {}
    gir_data = {}
    nonlinear_method_labels = []

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION & WELFARE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SIMULATION & WELFARE RESULTS")
    print("═" * 72)
    print(f"  Analysis: {config['analysis_name']}")
    print(f"  Model: {config['model_dir']}")
    print(f"  Experiments: {list(experiments_to_analyze.keys())}")
    print(f"  Sectors: {n_sectors}")
    print("─" * 72, flush=True)

    for experiment_label, exp_data in experiments_data.items():
        print(f"\n  ▶ {experiment_label}", flush=True)
        experiment_results = _run_experiment_analysis(
            experiment_label=experiment_label,
            exp_data=exp_data,
            save_dir=save_dir,
            nn_config_base=nn_config_base,
            econ_model=econ_model,
            nonlinear_simulation_runners=nonlinear_simulation_runners,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            stoch_ss_fn=stoch_ss_fn,
            stoch_ss_loss_fn=stoch_ss_loss_fn,
            gir_fn=gir_fn,
            config_dict=config,
            analysis_hooks=analysis_hooks,
        )

        raw_simulation_data.update(experiment_results["raw_simulation_data"])
        analysis_variables_data.update(experiment_results["analysis_variables"])
        welfare_costs.update(experiment_results["welfare_costs"])
        stochastic_ss_states.update(experiment_results["stochastic_ss_states"])
        stochastic_ss_policies.update(experiment_results["stochastic_ss_policies"])
        stochastic_ss_data.update(experiment_results["stochastic_ss_data"])
        stochastic_ss_loss.update(experiment_results["stochastic_ss_loss"])
        gir_data[experiment_label] = experiment_results["gir_data"]
        nonlinear_method_labels.extend(experiment_results.get("nonlinear_method_labels", []))

    # Add welfare costs from Dynare simulation methods (if available).
    dynare_welfare_inputs = {
        "FirstOrder": dynare_1st_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "SecondOrder": dynare_so_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "PerfectForesight": dynare_pf_artifact["active_simul"] if model_data_simulation_file is not None else None,
        "MITShocks": dynare_mit_artifact["active_simul"] if model_data_simulation_file is not None else None,
    }
    for method_name, simul_data in dynare_welfare_inputs.items():
        welfare_cost = _welfare_cost_from_dynare_simul(
            simul_data,
            method_name,
            state_ss,
            policies_ss,
            econ_model,
            welfare_fn,
            welfare_ss,
            config,
        )
        if welfare_cost is not None:
            welfare_costs[method_name] = welfare_cost
            print(f"    Welfare cost ({method_name}): {float(welfare_cost):.4f}%")

    theoretical_stats = {}
    matlab_ir_data = None
    upstreamness_data = None
    postprocess_context = None

    if analysis_hooks is not None and hasattr(analysis_hooks, "prepare_postprocess_analysis"):
        model_postprocess = analysis_hooks.prepare_postprocess_analysis(
            config=config,
            model_dir=model_dir,
            analysis_dir=analysis_dir,
            simulation_dir=simulation_dir,
            irs_dir=irs_dir,
            econ_model=econ_model,
            model_data=md,
            stats=stats,
            policies_ss=policies_ss,
            state_ss=state_ss,
            raw_simulation_data=raw_simulation_data,
            analysis_variables_data=analysis_variables_data,
            stochastic_ss_states=stochastic_ss_states,
            stochastic_ss_policies=stochastic_ss_policies,
            stochastic_ss_data=stochastic_ss_data,
            gir_data=gir_data,
            dynare_simulations=dynare_welfare_inputs,
            irs_path=irs_path,
        )
    else:
        model_postprocess = run_model_postprocess(
            analysis_hooks=analysis_hooks,
            config=config,
            model_dir=model_dir,
            analysis_dir=analysis_dir,
            simulation_dir=simulation_dir,
            irs_dir=irs_dir,
            econ_model=econ_model,
            model_data=md,
            stats=stats,
            policies_ss=policies_ss,
            state_ss=state_ss,
            raw_simulation_data=raw_simulation_data,
            analysis_variables_data=analysis_variables_data,
            stochastic_ss_states=stochastic_ss_states,
            stochastic_ss_policies=stochastic_ss_policies,
            stochastic_ss_data=stochastic_ss_data,
            gir_data=gir_data,
            dynare_simulations=dynare_welfare_inputs,
            irs_path=irs_path,
        )

    analysis_variables_data = model_postprocess.get("analysis_variables_data", analysis_variables_data)
    calibration_method_stats = model_postprocess.get("calibration_method_stats")
    theoretical_stats = model_postprocess.get("theoretical_stats", theoretical_stats)
    matlab_ir_data = model_postprocess.get("matlab_ir_data", matlab_ir_data)
    upstreamness_data = model_postprocess.get("upstreamness_data", upstreamness_data)
    postprocess_context = model_postprocess.get("postprocess_context")
    output_display_label_map = _build_output_display_label_map(config)
    display_postprocess_context = _apply_display_labels_to_postprocess_context(
        postprocess_context,
        output_display_label_map,
    )
    display_gir_data = _apply_display_labels_to_mapping(gir_data, output_display_label_map)
    display_stochastic_ss_states = _apply_display_labels_to_mapping(stochastic_ss_states, output_display_label_map)
    display_stochastic_ss_policies = _apply_display_labels_to_mapping(stochastic_ss_policies, output_display_label_map)
    display_stochastic_ss_data = _apply_display_labels_to_mapping(stochastic_ss_data, output_display_label_map)
    display_welfare_costs = _apply_display_labels_to_mapping(welfare_costs, output_display_label_map)
    display_raw_simulation_data = _apply_display_labels_to_mapping(raw_simulation_data, output_display_label_map)
    display_stochss_methods_to_include = cast(
        "list[str] | None",
        _apply_display_labels_to_sequence(
            config.get("stochss_methods_to_include"),
            output_display_label_map,
        ),
    )
    if not display_stochss_methods_to_include:
        display_stochss_methods_to_include = cast("list[str] | None", list(display_stochastic_ss_data.keys()))

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL VS DATA MOMENTS TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  MODEL VS DATA MOMENTS TABLE")
    print("═" * 72, flush=True)
    calibration_emp = (
        md.get("EmpiricalTargets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("empirical_targets")
        or (md.get("Calibration") or md.get("calibration") or {}).get("EmpiricalTargets")
    )
    fo_stats = stats.get("FirstOrder") or stats.get("firstorder")
    calibration_model_stats = (
        (fo_stats.get("ModelStats") or fo_stats.get("modelstats")) if isinstance(fo_stats, dict) else None
    )
    model_vs_data_aliases = {
        "FirstOrder": "1st",
        "Log-Linear": "1st",
        "LogLinear": "1st",
        "SecondOrder": "2nd",
        "Second-Order": "2nd",
        "PerfectForesight": "PF",
        "Perfect Foresight": "PF",
        "MITShock": "MITShocks",
        "MIT Shocks": "MITShocks",
        "MIT shocks": "MITShocks",
        "NonlinearCS": "Nonlinear-CS",
        "Nonlinear_CS": "Nonlinear-CS",
    }

    def _normalize_model_vs_data_method_name(method_name: str) -> str:
        return model_vs_data_aliases.get(method_name, method_name)

    model_vs_data_methods_cfg = config.get("model_vs_data_methods_to_include")
    filtered_calibration_method_stats = calibration_method_stats
    if model_vs_data_methods_cfg and calibration_method_stats is not None:
        if isinstance(model_vs_data_methods_cfg, str):
            model_vs_data_methods_cfg = [model_vs_data_methods_cfg]
        requested_methods = [_normalize_model_vs_data_method_name(name) for name in model_vs_data_methods_cfg]
        requested_method_set = set(requested_methods)
        filtered_calibration_method_stats = {
            method_name: stats_dict
            for method_name, stats_dict in calibration_method_stats.items()
            if method_name in requested_method_set
        }
        if filtered_calibration_method_stats:
            print(f"  Model-vs-data methods: {list(filtered_calibration_method_stats.keys())}")
        else:
            print(
                "  ⚠ model_vs_data_methods_to_include did not match any available methods; "
                f"using all available methods {list(calibration_method_stats.keys())}."
            )
            filtered_calibration_method_stats = calibration_method_stats
    if calibration_emp is not None:
        create_calibration_table(
            empirical_targets=calibration_emp,
            first_order_model_stats=calibration_model_stats,
            method_model_stats=filtered_calibration_method_stats,
            save_path=os.path.join(analysis_dir, "calibration_table.tex"),
            analysis_name=config["analysis_name"],
        )
    else:
        print(
            "  ⚠ Calibration table skipped: no empirical targets in ModelData (EmpiricalTargets / calibration.empirical_targets)."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATE IMPULSE RESPONSES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  AGGREGATE IMPULSE RESPONSES")
    print("═" * 72, flush=True)
    if analysis_hooks is not None and hasattr(analysis_hooks, "render_aggregate_ir_outputs") and postprocess_context:
        analysis_hooks.render_aggregate_ir_outputs(
            config=config,
            irs_dir=irs_dir,
            econ_model=econ_model,
            gir_data=display_gir_data,
            postprocess_context=display_postprocess_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTORAL VARIABLES IN STOCHASTIC SS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SECTORAL VARIABLES IN STOCHASTIC SS")
    print("═" * 72, flush=True)
    if (
        analysis_hooks is not None
        and hasattr(analysis_hooks, "render_sectoral_stochss_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_sectoral_stochss_outputs(
            config=config,
            simulation_dir=simulation_dir,
            econ_model=econ_model,
            stochastic_ss_states=display_stochastic_ss_states,
            stochastic_ss_policies=display_stochastic_ss_policies,
            postprocess_context=display_postprocess_context,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # AGGREGATE STOCHASTIC STEADY STATE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  AGGREGATE STOCHASTIC STEADY STATE")
    print("═" * 72, flush=True)
    create_stochastic_ss_aggregates_table(
        stochastic_ss_data=display_stochastic_ss_data,
        save_path=os.path.join(analysis_dir, "stochastic_ss_aggregates_table.tex"),
        analysis_name=config["analysis_name"],
        methods_to_include=display_stochss_methods_to_include,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # DESCRIPTIVE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  DESCRIPTIVE STATISTICS")
    print("═" * 72, flush=True)
    method_aliases = {
        "FirstOrder": "Log-Linear",
        "LogLinear": "Log-Linear",
        "Second-Order": "SecondOrder",
        "Perfect Foresight": "PerfectForesight",
        "MIT Shocks": "MITShocks",
        "MIT shocks": "MITShocks",
        "MITShock": "MITShocks",
    }

    def _normalize_method_name(method_name: str) -> str:
        return method_aliases.get(method_name, method_name)

    available_methods = sorted({_normalize_method_name(k) for k in analysis_variables_data.keys()})
    print(f"  Available methods (canonical): {available_methods}")

    benchmark_methods_cfg = config.get("ergodic_methods_to_include")
    benchmark_methods = None
    if benchmark_methods_cfg:
        if isinstance(benchmark_methods_cfg, str):
            benchmark_methods_cfg = [benchmark_methods_cfg]
        benchmark_methods = {_normalize_method_name(m) for m in benchmark_methods_cfg}

    selected_methods = set(available_methods) if benchmark_methods is None else set(benchmark_methods)
    if config.get("always_include_nonlinear_methods", True):
        selected_methods.update(_normalize_method_name(label) for label in nonlinear_method_labels)

    filtered_analysis_variables_data = {
        _normalize_method_name(k): v
        for k, v in analysis_variables_data.items()
        if _normalize_method_name(k) in selected_methods
    }
    desc_vars = config.get("descriptive_stats_variables")
    if desc_vars:
        desc_analysis_data = {
            method: {k: v for k, v in variables.items() if k in desc_vars}
            for method, variables in filtered_analysis_variables_data.items()
        }
        desc_analysis_data = {k: v for k, v in desc_analysis_data.items() if v}
    else:
        desc_analysis_data = filtered_analysis_variables_data

    display_filtered_analysis_variables_data = _apply_display_labels_to_mapping(
        filtered_analysis_variables_data,
        output_display_label_map,
    )
    display_desc_analysis_data = _apply_display_labels_to_mapping(
        desc_analysis_data,
        output_display_label_map,
    )

    create_descriptive_stats_table(
        analysis_variables_data=display_desc_analysis_data,
        save_path=os.path.join(simulation_dir, "descriptive_stats_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # WELFARE COSTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  WELFARE COSTS")
    print("═" * 72, flush=True)
    create_welfare_table(
        welfare_data=display_welfare_costs,
        save_path=os.path.join(analysis_dir, "welfare_table.tex"),
        analysis_name=config["analysis_name"],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTORAL IMPULSE RESPONSES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("  SECTORAL IMPULSE RESPONSES")
    print("═" * 72, flush=True)
    if analysis_hooks is not None and hasattr(analysis_hooks, "render_sectoral_ir_outputs") and postprocess_context:
        analysis_hooks.render_sectoral_ir_outputs(
            config=config,
            irs_dir=irs_dir,
            econ_model=econ_model,
            gir_data=display_gir_data,
            postprocess_context=display_postprocess_context,
        )

    if (
        analysis_hooks is not None
        and hasattr(analysis_hooks, "render_ergodic_sectoral_outputs")
        and postprocess_context
    ):
        analysis_hooks.render_ergodic_sectoral_outputs(
            config=config,
            simulation_dir=simulation_dir,
            econ_model=econ_model,
            raw_simulation_data=display_raw_simulation_data,
            postprocess_context=display_postprocess_context,
        )

    # Model-specific plots
    if MODEL_SPECIFIC_PLOTS:
        for plot_spec in MODEL_SPECIFIC_PLOTS:
            plot_name = plot_spec["name"]
            plot_function = plot_spec["function"]

            for experiment_label, sim_data in raw_simulation_data.items():
                if sim_data.get("simulation_kind", "ergodic") != "ergodic":
                    continue
                try:
                    plot_function(
                        simul_obs=sim_data["simul_obs"],
                        simul_policies=sim_data["simul_policies"],
                        simul_analysis_variables=sim_data["simul_analysis_variables"],
                        save_path=os.path.join(simulation_dir, f"{plot_name}_{experiment_label}.png"),
                        analysis_name=config["analysis_name"],
                        econ_model=econ_model,
                        experiment_label=experiment_label,
                    )
                except Exception as e:
                    print(f"    ✗ Failed to create {plot_name} for {experiment_label}: {e}", flush=True)

    _write_analysis_results_latex(
        config_dict=config,
        analysis_dir=analysis_dir,
        simulation_dir=simulation_dir,
        irs_dir=irs_dir,
        econ_model=econ_model,
    )

    print("\n" + "═" * 72)
    print("  ANALYSIS COMPLETE")
    print("═" * 72, flush=True)

    return {
        "analysis_variables_data": analysis_variables_data,
        "raw_simulation_data": raw_simulation_data,
        "welfare_costs": welfare_costs,
        "stochastic_ss_data": stochastic_ss_data,
        "stochastic_ss_states": stochastic_ss_states,
        "stochastic_ss_policies": stochastic_ss_policies,
        "stochastic_ss_loss": stochastic_ss_loss,
        "gir_data": gir_data,
        "matlab_ir_data": matlab_ir_data,
        "upstreamness_data": upstreamness_data,
        "theoretical_stats": theoretical_stats,  # Pre-computed stats for log-linear and perfect foresight
    }


# Legacy long-ergodic simulation branch kept for later reuse:
# simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config))
# simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
#     train_state,
#     econ_model,
#     config,
#     simulation_fn,
#     analysis_hooks=analysis_hooks,
# )
# simul_obs_full = simul_obs
# simul_policies_full = simul_policies


if __name__ == "__main__":
    main()
