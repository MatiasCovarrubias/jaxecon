"""
RbcProdNet-specific plotting helpers copied from the shared analysis layer.

These helpers remain local to the model so sector indexing, MATLAB benchmark handling,
and upstreamness-based visualizations do not leak back into `DEQN/analysis/`.
"""

import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

from DEQN.analysis.shock_keys import build_shock_key, format_shock_size_token, parse_shock_size_token

# Plot styling configuration
sns.set_style("whitegrid")
palette = "dark"
sns.set_palette(palette)
colors = sns.color_palette(palette, 10)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16
DEFAULT_IR_BENCHMARK_METHODS = ["PerfectForesight", "FirstOrder"]

# Set font family and sizes globally
plt.rc("font", family="sans-serif", size=SMALL_SIZE)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

plt.rc("axes", titlesize=LARGE_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=LARGE_SIZE)

plt.rc("mathtext", fontset="dejavusans")

LINESTYLES_RANKED = ["-", "--", "-.", ":"]
MARKERS_RANKED = [None, None, None, None]


def _ranked_linestyle(rank: int) -> str:
    return LINESTYLES_RANKED[rank % len(LINESTYLES_RANKED)]


def _ranked_marker(rank: int):
    return MARKERS_RANKED[rank % len(MARKERS_RANKED)]


def _ranked_color(rank: int):
    return colors[rank % len(colors)]


def _benchmark_style(benchmark_rank: int = 0) -> dict[str, Any]:
    benchmark_colors = ["black", "dimgray", "gray", "silver"]
    return {
        "color": benchmark_colors[benchmark_rank % len(benchmark_colors)],
        "linewidth": 2.0,
        "linestyle": _ranked_linestyle(benchmark_rank + 1),
        "marker": _ranked_marker(benchmark_rank + 1),
        "alpha": 0.9,
    }


def _experiment_style(experiment_rank: int, response_kind: str = "IR_stoch_ss") -> dict[str, Any]:
    del response_kind
    return {
        "color": _ranked_color(experiment_rank),
        "linewidth": 2.5,
        "linestyle": "-",
        "marker": None,
        "alpha": 0.9,
    }


def _escape_latex(text: str) -> str:
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
    return "".join(replacements.get(char, char) for char in text)


def _format_number_list(values: list[int]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return str(values[0])
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(str(value) for value in values[:-1]) + f", and {values[-1]}"


def _join_text_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _describe_response_source(response_source: str) -> str:
    if response_source == "GIR":
        return (
            "The solid DEQN line reports the generalized impulse response, averaging the difference "
            "between a zero-shock path and a path whose TFP state is shifted on impact across draws "
            "from the ergodic distribution."
        )
    return (
        "The solid DEQN line reports the stochastic-steady-state impulse response, comparing "
        "shocked and unshocked paths that start from the stochastic steady state."
    )


def _describe_benchmark_method(benchmark_method: str) -> str:
    benchmark_labels = {
        "FirstOrder": "1st-order approximation",
        "SecondOrder": "2nd-order approximation",
        "PerfectForesight": "perfect foresight",
        "MITShocks": "MIT shocks",
    }
    return benchmark_labels.get(benchmark_method, benchmark_method)


def _resolve_ir_benchmark_methods(
    benchmark_methods: Optional[list[str]] = None, benchmark_method: Optional[str] = None
) -> list[str]:
    if benchmark_methods is None:
        benchmark_methods = [benchmark_method] if benchmark_method else list(DEFAULT_IR_BENCHMARK_METHODS)
    elif isinstance(benchmark_methods, str):
        benchmark_methods = [benchmark_methods]

    resolved_methods = []
    for method in benchmark_methods:
        if method and method not in resolved_methods:
            resolved_methods.append(method)
    return resolved_methods or list(DEFAULT_IR_BENCHMARK_METHODS)


def _write_figure_note_tex(figure_path: str, note_text: str) -> None:
    note_path = os.path.splitext(figure_path)[0] + "_note.tex"
    note_tex = (
        r"\begin{minipage}{0.92\textwidth}" + "\n"
        r"\footnotesize" + "\n"
        r"\textit{Notes:} "
        + _escape_latex(note_text)
        + "\n"
        + r"\end{minipage}"
        + "\n"
    )
    with open(note_path, "w") as note_file:
        note_file.write(note_tex)


def _print_saved_file(path: str, indent: str = "    ") -> None:
    print(f"{indent}Saved: {os.path.basename(path)}", flush=True)


def _build_ir_note(
    *,
    variable_to_plot: str,
    sector_label: str,
    shock_sizes: list[int],
    benchmark_methods: list[str],
    response_source: str,
    negative_only: bool,
    is_aggregate: bool,
) -> str:
    shock_text = _format_number_list(shock_sizes)
    if negative_only:
        layout_text = (
            f"The figure shows the response to a negative {shock_text} percent TFP shock to {sector_label}."
            if shock_text
            else f"The figure shows a negative TFP shock to {sector_label}."
        )
    else:
        layout_text = (
            f"Each row corresponds to a {shock_text} percent TFP shock to {sector_label}; "
            "the left column shows negative shocks and the right column positive shocks."
            if shock_text
            else f"The figure compares positive and negative TFP shocks to {sector_label}."
        )

    anchor_text = "Dashed comparison IRs are anchored at the deterministic steady state."
    axis_text = (
        "The horizontal axis reports periods after impact. The vertical axis reports impulse responses in percent."
    )
    if variable_to_plot == "gammaij_client":
        axis_text = (
            "The horizontal axis reports periods after impact. The vertical axis reports deviations in the client "
            "sector expenditure share, so this panel should be read as a share response rather than a log-percent response."
        )

    described_benchmarks = [_describe_benchmark_method(method) for method in benchmark_methods]
    if not described_benchmarks:
        benchmark_text = ""
    elif len(described_benchmarks) == 1:
        benchmark_text = f"The dashed line reports the {described_benchmarks[0]}."
    else:
        benchmark_text = f"The dashed lines report {_join_text_list(described_benchmarks)}."
    if not is_aggregate and "_client" in variable_to_plot:
        benchmark_text += (
            " Variables with the suffix 'client' refer to the petroleum client sector of the shocked sector in the "
            "input-output network."
        )
    elif not is_aggregate:
        benchmark_text += " Un-suffixed sectoral variables refer to the shocked sector itself."

    return " ".join(
        part
        for part in [
            layout_text,
            _describe_response_source(response_source),
            anchor_text,
            axis_text,
            benchmark_text,
        ]
        if part
    )


def _build_sectoral_distribution_note(
    *,
    variable_title: str,
    display_labels: list[str],
    source_kind: str,
    include_upstreamness: bool,
) -> str:
    comparison_text = "Bars report the displayed experiment sector by sector."
    if source_kind == "stochss":
        source_text = (
            "The figure reports the stochastic steady state computed by taking draws from the ergodic distribution, "
            "simulating forward with zero shocks, and taking the common limit to which those paths converge; "
            "convergence to the same point across initial draws is checked."
        )
    else:
        source_text = (
            "The figure reports the ergodic mean from the long nonlinear simulation, i.e. the time average over "
            "the simulated ergodic sample."
        )

    upstreamness_text = ""
    if include_upstreamness:
        upstreamness_text = (
            " The textbox reports cross-sector correlations with IO upstreamness and investment upstreamness."
        )

    return (
        f"{source_text} {comparison_text} The horizontal axis lists sectors sorted by the displayed "
        f"experiment from highest to lowest {variable_title.lower()}. The vertical axis reports log differences "
        "from the deterministic steady state; for small changes, a value of -0.1 means roughly 0.1 percent below "
        f"the deterministic steady state.{upstreamness_text}"
    )


def _single_experiment_name(data: Dict[str, Any], context: str) -> str:
    names = list(data.keys()) if data else []
    if len(names) != 1:
        raise ValueError(f"{context} expects exactly one experiment; got {names}.")
    return names[0]


def plot_ergodic_histograms(
    analysis_variables_data: Dict[str, Any],
    figsize: Tuple[float, float] = (15, 10),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    theo_dist_params: Optional[Dict[str, Dict[str, float]]] = None,
    benchmark_order: Optional[list[str]] = None,
):
    """
    Create publication-quality histograms of ergodic distributions for analysis variables.

    Parameters:
    -----------
    analysis_variables_data : dict
        Dictionary where keys are method names and values are dictionaries mapping
        variable labels to arrays of values: {method_name: {var_label: array}}
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename. If None, no analysis name is added.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.
    theo_dist_params : dict, optional
        Theoretical distribution parameters for methods that should be plotted as
        smooth PDF curves instead of histograms. Format:
        {method_name: {var_label: {"mean": float, "std": float}}}
        Methods in this dict will show a smooth normal PDF curve.
    benchmark_order : list[str], optional
        Ordered benchmark labels that should use the same benchmark styling
        convention as the aggregate IR figures.

    Returns:
    --------
    list of (fig, ax) tuples for each analysis variable
    """
    # Get method names from both simulated and theoretical sources.
    method_names = list(analysis_variables_data.keys())
    if theo_dist_params is not None:
        for method_name in theo_dist_params.keys():
            if method_name not in method_names:
                method_names.append(method_name)

    # Extract variable labels that exist in all methods once theoretical overlays are accounted for.
    excluded_vars = []
    candidate_vars = []
    for exp_name, exp_data in analysis_variables_data.items():
        candidate_vars.extend(v for v in exp_data.keys() if v not in excluded_vars)
    if theo_dist_params is not None:
        for exp_params in theo_dist_params.values():
            candidate_vars.extend(v for v in exp_params.keys() if v not in excluded_vars)
    candidate_vars = list(dict.fromkeys(candidate_vars))

    # Only include variables that exist in all experiments
    var_labels = []
    for var in candidate_vars:
        exists_in_all = True
        missing_in = []
        for method_name in method_names:
            in_simulation = method_name in analysis_variables_data and var in analysis_variables_data[method_name]
            in_theory = theo_dist_params is not None and var in theo_dist_params.get(method_name, {})
            if not (in_simulation or in_theory):
                exists_in_all = False
                missing_in.append(method_name)
        if exists_in_all:
            var_labels.append(var)
        else:
            print(f"  Note: Skipping '{var}' - not available in: {missing_in}")

    # Create safe file names from labels
    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    var_filenames = [make_safe_filename(label) for label in var_labels]

    figures = []

    # Track which methods have theoretical params for this function call.
    theo_methods = set()
    if theo_dist_params is not None:
        theo_methods = set(theo_dist_params.keys())
    benchmark_order = benchmark_order or []
    benchmark_style_map = {
        label: _benchmark_style(rank) for rank, label in enumerate(benchmark_order)
    }

    for var_label, var_filename in zip(var_labels, var_filenames):
        # Use a tighter fixed range and slightly fewer bins to reduce visual noise.
        bin_range = (-7.5, 7.5)

        # Create bins using the fixed range
        bins = np.linspace(bin_range[0], bin_range[1], 21)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        # For theoretical curves, use a smooth x range
        x_smooth = np.linspace(bin_range[0], bin_range[1], 200)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

        # Plot one line for each method.
        for i, method_name in enumerate(method_names):
            # Check if this method should use theoretical PDF.
            use_theo = (
                method_name in theo_methods
                and theo_dist_params is not None
                and var_label in theo_dist_params.get(method_name, {})
            )

            if use_theo:
                # Plot smooth theoretical normal PDF
                params = theo_dist_params[method_name][var_label]
                mean_pct = params["mean"] * 100  # Convert to percentage
                std_pct = params["std"] * 100  # Convert to percentage
                if std_pct <= 0:
                    continue

                # Compute PDF and scale to match histogram frequencies
                # PDF integrated over bin_width gives probability per bin
                pdf_values = norm.pdf(x_smooth, loc=mean_pct, scale=std_pct) * bin_width

                style = benchmark_style_map.get(method_name, _experiment_style(i, "IR_stoch_ss"))
                ax.plot(x_smooth, pdf_values, label=method_name, **style)
            else:
                if method_name not in analysis_variables_data or var_label not in analysis_variables_data[method_name]:
                    continue
                var_values = np.asarray(analysis_variables_data[method_name][var_label], dtype=float) * 100
                if var_values.size == 0:
                    continue
                # Calculate histogram from samples
                counts, _ = np.histogram(var_values, bins=bins)
                freqs = counts / len(var_values)

                # Plot the frequency line
                style = benchmark_style_map.get(method_name, _experiment_style(i, "IR_stoch_ss"))
                ax.plot(bin_centers, freqs, label=method_name, **style)

        # Add vertical line at deterministic steady state (x=0)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Deterministic SS", alpha=0.7)

        # Styling
        ax.set_xlabel(
            f"{var_label} (% deviations from deterministic SS)",
            fontweight="bold",
            fontsize=MEDIUM_SIZE,
        )
        ax.set_ylabel("Frequency", fontweight="bold", fontsize=MEDIUM_SIZE)

        # Legend
        ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

        # Grid
        ax.grid(True, alpha=0.3)

        # Apply consistent styling
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
        ax.set_xlim(*bin_range)

        # Adjust layout
        plt.tight_layout()

        # Save if directory provided
        if save_dir:
            # Create filename with analysis name if provided
            if analysis_name:
                filename = f"Histogram_{var_filename}_{analysis_name}.png"
            else:
                filename = f"Histogram_{var_filename}_comparative.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
            _print_saved_file(save_path)

        figures.append((fig, ax))

    plt.show()

    return figures


def plot_gir_responses(
    gir_data: Dict[str, Any],
    states_to_plot: Optional[list] = None,
    variables_to_plot: Optional[list] = None,
    shock_config: str = "neg_20",
    figsize: Tuple[float, float] = (12, 8),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create publication-quality plots of Generalized Impulse Responses over time.
    Each state gets its own separate plot for each analysis variable.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from analysis.
        New structure: {experiment_name: {state_name: {"state_idx": int, "pos_5": {...}, "neg_20": {...}, ...}}}
    states_to_plot : list, optional
        Which states to plot. If None, plots all states.
    variables_to_plot : list, optional
        Which variable labels to plot. If None, plots all variables.
    shock_config : str
        Which shock configuration to plot (e.g., "neg_20", "pos_10"). Default is "neg_20".
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    list of (fig, ax) tuples for each state-analysis variable combination
    """
    # Get the single experiment to determine states and variables.
    first_experiment = _single_experiment_name(gir_data, "plot_gir_responses")
    first_exp_data = gir_data[first_experiment]

    # Get state names
    all_state_names = list(first_exp_data.keys())
    if states_to_plot is None:
        # Default to all states
        states_to_plot = all_state_names
    else:
        # Filter to requested states
        states_to_plot = [s for s in states_to_plot if s in all_state_names]

    # Get variable labels from first state's GIR data (using specified shock config)
    first_state = states_to_plot[0]
    first_state_data = first_exp_data[first_state]

    # Handle new data structure with shock configs like "pos_5", "neg_20", etc.
    if shock_config in first_state_data and "gir_analysis_variables" in first_state_data[shock_config]:
        gir_vars_dict = first_state_data[shock_config]["gir_analysis_variables"]
    elif "gir_analysis_variables" in first_state_data:
        # Old data structure (backward compatibility)
        gir_vars_dict = first_state_data["gir_analysis_variables"]
        shock_config = None  # Signal to use old structure
    else:
        # Try to find any shock config
        available_configs = [k for k in first_state_data.keys() if k not in ["state_idx"]]
        if available_configs:
            shock_config = available_configs[0]
            gir_vars_dict = first_state_data[shock_config]["gir_analysis_variables"]
        else:
            print(f"Warning: No GIR data found for state {first_state}")
            return []

    all_var_labels = list(gir_vars_dict.keys())

    if variables_to_plot is None:
        variables_to_plot = all_var_labels
    else:
        # Filter to requested variables
        variables_to_plot = [v for v in variables_to_plot if v in all_var_labels]

    # Get time length from first variable
    first_var = variables_to_plot[0]
    time_length = len(gir_vars_dict[first_var])
    time_periods = np.arange(time_length)

    # Use colors from the global palette
    plot_colors = colors

    # Create safe file names from labels
    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    figures = []

    # Loop through each state first, then each analysis variable
    for state_name in states_to_plot:
        for var_label in variables_to_plot:
            # Create figure for this state-analysis variable combination
            fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

            state_data = gir_data[first_experiment][state_name]

            # Get GIR data based on structure
            if shock_config and shock_config in state_data:
                gir_analysis_variables = state_data[shock_config]["gir_analysis_variables"]
            elif "gir_analysis_variables" in state_data:
                gir_analysis_variables = state_data["gir_analysis_variables"]
            else:
                continue

            if var_label not in gir_analysis_variables:
                continue

            # Convert to percentages
            response_pct = gir_analysis_variables[var_label] * 100

            label = f"{state_name} Response"
            style = _experiment_style(0, "IR_stoch_ss")

            # Plot the impulse response
            ax.plot(
                time_periods[: len(response_pct)],
                response_pct,
                label=label,
                **style,
            )

            # Add horizontal line at zero
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

            # Styling
            ax.set_xlabel("Time Periods", fontweight="bold", fontsize=MEDIUM_SIZE)
            ax.set_ylabel(f"{var_label} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

            # Grid
            ax.grid(True, alpha=0.3)

            # Apply consistent styling
            ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

            # Set x-axis to start from 0
            ax.set_xlim(0, time_length - 1)

            # Adjust layout
            plt.tight_layout()

            # Save if directory provided
            if save_dir:
                # Create filename with state and analysis variable names
                safe_state_name = make_safe_filename(state_name)
                safe_var_name = make_safe_filename(var_label)
                shock_suffix = f"_{shock_config}" if shock_config else ""
                if analysis_name:
                    filename = f"GIR_{safe_var_name}_{safe_state_name}{shock_suffix}_{analysis_name}.png"
                else:
                    filename = f"GIR_{safe_var_name}_{safe_state_name}{shock_suffix}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
                _print_saved_file(save_path)

            figures.append((fig, ax))

    plt.show()

    return figures


def plot_combined_impulse_responses(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sectors_to_plot: list,
    sector_labels: list,
    variables_to_plot: Optional[list] = None,
    shock_sizes_to_plot: list = [20],
    figsize: Tuple[float, float] = (14, 10),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
):
    """
    Create combined plots showing GIR responses alongside MATLAB perfect foresight and loglinear IRs.
    Shows both positive and negative shocks on the same graph for each shock size.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from JAX analysis
        Structure: {experiment_name: {state_name: {"gir_analysis_variables": {var_label: array}, "state_idx": int}}}
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs
    sectors_to_plot : list
        List of sector indices (0-based) to plot
    sector_labels : list
        List of sector labels for display
    variables_to_plot : list, optional
        Which variable labels to plot. If None, plots all variables.
    shock_sizes_to_plot : list
        List of shock sizes to include in plots (e.g., [5, 10, 20])
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_dir : str, optional
        Directory to save plots to. If None, plots are not saved.
    analysis_name : str, optional
        Name of the analysis to include in the filename.
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.
    max_periods : int
        Maximum number of periods to plot

    Returns:
    --------
    list of (fig, ax) tuples for each sector-variable-shock_size combination
    """
    from DEQN.analysis.matlab_irs import get_matlab_ir_for_analysis_variable

    first_experiment = _single_experiment_name(gir_data, "plot_combined_impulse_responses")
    first_exp_data = gir_data[first_experiment]

    all_state_names = list(first_exp_data.keys())

    first_state = all_state_names[0]
    gir_vars_dict = first_exp_data[first_state]["gir_analysis_variables"]
    all_var_labels = list(gir_vars_dict.keys())

    if variables_to_plot is None:
        variables_to_plot = all_var_labels
    else:
        variables_to_plot = [v for v in variables_to_plot if v in all_var_labels]

    time_periods = np.arange(max_periods)

    def make_safe_filename(label):
        return label.replace(" ", "_").replace(".", "").replace("/", "_")

    figures = []

    n_sectors = len(sector_labels)

    for sector_idx in sectors_to_plot:
        sector_label = sector_labels[sector_idx] if sector_idx < len(sector_labels) else f"Sector {sector_idx + 1}"

        state_name = None
        for candidate in all_state_names:
            gir_info = first_exp_data[candidate]
            candidate_state_idx = gir_info.get("state_idx")
            if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors + sector_idx:
                state_name = candidate
                break

        if state_name is None:
            for possible_idx in [sector_idx, n_sectors + sector_idx]:
                possible_name = f"state_{possible_idx}"
                if possible_name in all_state_names:
                    state_name = possible_name
                    break

        if state_name is None:
            print(f"Warning: No GIR data found for sector {sector_idx}")
            print(f"  Available states: {all_state_names}")
            continue

        for var_label in variables_to_plot:
            for shock_size in shock_sizes_to_plot:
                fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

                pos_key = build_shock_key("pos", shock_size)
                neg_key = build_shock_key("neg", shock_size)

                matlab_irs_pos = get_matlab_ir_for_analysis_variable(matlab_ir_data, sector_idx, var_label, max_periods)
                matlab_irs_neg = get_matlab_ir_for_analysis_variable(matlab_ir_data, sector_idx, var_label, max_periods)

                pos_keys = (
                    [pos_key]
                    if (matlab_irs_pos and pos_key in matlab_irs_pos)
                    else (list(matlab_irs_pos.keys()) if matlab_irs_pos else [])
                )
                neg_keys = (
                    [neg_key]
                    if (matlab_irs_neg and neg_key in matlab_irs_neg)
                    else (list(matlab_irs_neg.keys()) if matlab_irs_neg else [])
                )

                for pk in pos_keys:
                    pos_loglin = matlab_irs_pos[pk]["loglin"][:max_periods] * 100
                    pct = pk.replace("pos_", "")
                    t_loglin = np.arange(len(pos_loglin))
                    ax.plot(t_loglin, pos_loglin, label=f"Loglinear (+{pct}%)", **_benchmark_style(0))
                    pos_determ = matlab_irs_pos[pk].get("determ")
                    if pos_determ is not None:
                        pos_determ = pos_determ[:max_periods] * 100
                        ax.plot(
                            np.arange(len(pos_determ)),
                            pos_determ,
                            label=f"Perfect Foresight (+{pct}%)",
                            **_benchmark_style(1),
                        )

                for nk in neg_keys:
                    neg_loglin = matlab_irs_neg[nk]["loglin"][:max_periods] * 100
                    pct = nk.replace("neg_", "")
                    t_loglin = np.arange(len(neg_loglin))
                    ax.plot(t_loglin, neg_loglin, label=f"Loglinear (-{pct}%)", **_benchmark_style(0))
                    neg_determ = matlab_irs_neg[nk].get("determ")
                    if neg_determ is not None:
                        neg_determ = neg_determ[:max_periods] * 100
                        ax.plot(
                            np.arange(len(neg_determ)),
                            neg_determ,
                            label=f"Perfect Foresight (-{pct}%)",
                            **_benchmark_style(1),
                        )

                if state_name in gir_data[first_experiment]:
                    gir_analysis_variables = gir_data[first_experiment][state_name]["gir_analysis_variables"]
                    if var_label in gir_analysis_variables:
                        response_pct = gir_analysis_variables[var_label][:max_periods] * 100
                        ax.plot(
                            time_periods[: len(response_pct)],
                            response_pct,
                            label="GIR",
                            **_experiment_style(0, "GIR"),
                        )

                ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

                ax.set_xlabel("Time Periods", fontweight="bold", fontsize=MEDIUM_SIZE)
                ax.set_ylabel(f"{var_label} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)
                ax.set_title(
                    f"{sector_label}: {var_label} Response ({shock_size}% shock)",
                    fontweight="bold",
                    fontsize=LARGE_SIZE,
                )

                ax.legend(frameon=True, framealpha=0.9, loc="best", fontsize=SMALL_SIZE - 1, ncol=2)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
                ax.set_xlim(0, max_periods - 1)

                plt.tight_layout()

                if save_dir:
                    safe_sector = make_safe_filename(sector_label)
                    safe_var = make_safe_filename(var_label)
                    if analysis_name:
                        filename = f"CombinedIR_{safe_var}_{safe_sector}_shock{shock_size}_{analysis_name}.png"
                    else:
                        filename = f"CombinedIR_{safe_var}_{safe_sector}_shock{shock_size}.png"
                    save_path = os.path.join(save_dir, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
                    _print_saved_file(save_path)

                figures.append((fig, ax))

    plt.show()

    return figures


def plot_ir_comparison_panel(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sector_idx: int,
    sector_label: str,
    variables_to_plot: list,
    shock_sizes: list = [5, 10, 20],
    figsize: Tuple[float, float] = (18, 12),
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
):
    """
    Create a panel figure showing all shock sizes for a single sector.
    Each row is a variable, each column is a shock size, with pos/neg on same plot.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from JAX analysis
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs
    sector_idx : int
        Sector index (0-based) to plot
    sector_label : str
        Label for the sector
    variables_to_plot : list
        List of variable labels to plot
    shock_sizes : list
        List of shock sizes (e.g., [5, 10, 20])
    figsize : tuple
        Figure size
    save_dir : str, optional
        Directory to save the figure
    analysis_name : str, optional
        Name for the analysis
    display_dpi : int
        DPI for display
    max_periods : int
        Maximum periods to plot

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    from DEQN.analysis.matlab_irs import get_matlab_ir_for_analysis_variable

    n_vars = len(variables_to_plot)
    n_sizes = len(shock_sizes)

    fig, axes = plt.subplots(n_vars, n_sizes, figsize=figsize, dpi=display_dpi, squeeze=False)

    first_experiment = _single_experiment_name(gir_data, "plot_sector_ir_matrix")
    first_exp_data = gir_data[first_experiment]
    all_state_names = list(first_exp_data.keys())

    n_sectors_in_model = 37

    state_name = None
    for candidate in all_state_names:
        gir_info = first_exp_data[candidate]
        candidate_state_idx = gir_info.get("state_idx")
        if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors_in_model + sector_idx:
            state_name = candidate
            break

    if state_name is None:
        for possible_idx in [sector_idx, n_sectors_in_model + sector_idx]:
            possible_name = f"state_{possible_idx}"
            if possible_name in all_state_names:
                state_name = possible_name
                break

    if state_name is None and all_state_names:
        print(f"Warning: No exact match for sector {sector_idx}, using first available state")
        state_name = all_state_names[0]

    time_periods = np.arange(max_periods)

    for i, var_label in enumerate(variables_to_plot):
        for j, shock_size in enumerate(shock_sizes):
            ax = axes[i, j]

            pos_key = build_shock_key("pos", shock_size)
            neg_key = build_shock_key("neg", shock_size)

            matlab_irs = get_matlab_ir_for_analysis_variable(matlab_ir_data, sector_idx, var_label, max_periods)

            if matlab_irs:
                if pos_key in matlab_irs:
                    pos_loglin = matlab_irs[pos_key]["loglin"][:max_periods] * 100
                    pos_determ = matlab_irs[pos_key]["determ"][:max_periods] * 100
                    ax.plot(
                        np.arange(len(pos_loglin)),
                        pos_loglin,
                        color=colors[4],
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.7,
                    )
                    ax.plot(
                        np.arange(len(pos_determ)),
                        pos_determ,
                        color=colors[2],
                        linewidth=1.5,
                        linestyle="-.",
                        alpha=0.7,
                    )

                if neg_key in matlab_irs:
                    neg_loglin = matlab_irs[neg_key]["loglin"][:max_periods] * 100
                    neg_determ = matlab_irs[neg_key]["determ"][:max_periods] * 100
                    ax.plot(
                        np.arange(len(neg_loglin)),
                        neg_loglin,
                        color=colors[5],
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.7,
                    )
                    ax.plot(
                        np.arange(len(neg_determ)),
                        neg_determ,
                        color=colors[3],
                        linewidth=1.5,
                        linestyle="-.",
                        alpha=0.7,
                    )

            if state_name in gir_data[first_experiment]:
                gir_vars = gir_data[first_experiment][state_name]["gir_analysis_variables"]
                if var_label in gir_vars:
                    response = gir_vars[var_label][:max_periods] * 100
                    ax.plot(
                        time_periods[: len(response)],
                        response,
                        color=colors[0],
                        linewidth=2,
                        alpha=0.9,
                    )

            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.grid(True, alpha=0.2)

            if i == 0:
                ax.set_title(f"±{shock_size}% shock", fontweight="bold", fontsize=MEDIUM_SIZE)
            if j == 0:
                ax.set_ylabel(var_label, fontweight="bold", fontsize=SMALL_SIZE)
            if i == n_vars - 1:
                ax.set_xlabel("Periods", fontsize=SMALL_SIZE)

            ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE - 2)
            ax.set_xlim(0, max_periods - 1)

    fig.suptitle(f"{sector_label}: Impulse Response Comparison", fontweight="bold", fontsize=LARGE_SIZE, y=1.02)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors[4], linewidth=1.5, linestyle="--", label="Loglinear (+)"),
        Line2D([0], [0], color=colors[5], linewidth=1.5, linestyle="--", label="Loglinear (-)"),
        Line2D([0], [0], color=colors[2], linewidth=1.5, linestyle="-.", label="Perfect Foresight (+)"),
        Line2D([0], [0], color=colors[3], linewidth=1.5, linestyle="-.", label="Perfect Foresight (-)"),
    ]

    legend_elements.append(Line2D([0], [0], color=colors[0], linewidth=2, label="GIR"))

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=SMALL_SIZE - 1,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()

    if save_dir:

        def make_safe_filename(label):
            return label.replace(" ", "_").replace(".", "").replace("/", "_")

        safe_sector = make_safe_filename(sector_label)
        if analysis_name:
            filename = f"IRPanel_{safe_sector}_{analysis_name}.png"
        else:
            filename = f"IRPanel_{safe_sector}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
        _print_saved_file(save_path)

    return fig, axes


def plot_sector_ir_by_shock_size(
    gir_data: Dict[str, Any],
    matlab_ir_data: Dict[str, Any],
    sector_idx: int,
    sector_label: str,
    variable_to_plot: str = "Agg. Consumption",
    shock_sizes: list = [12.5, 50],
    figsize: Optional[Tuple[float, float]] = None,
    save_dir: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
    max_periods: int = 80,
    n_sectors: int = 37,
    benchmark_method: Optional[str] = None,
    benchmark_methods: Optional[list[str]] = None,
    response_source: str = "IR_stoch_ss",
    agg_consumption_mode: bool = False,
    negative_only: bool = False,
    filename_suffix: Optional[str] = None,
    policies_ss: Optional[np.ndarray] = None,
    P_ergodic: Optional[np.ndarray] = None,
    Pk_ergodic: Optional[np.ndarray] = None,
    state_ss: Optional[np.ndarray] = None,
    ergodic_price_aggregation: bool = False,
):
    """
    Create a figure with one row per shock size and two columns (negative / positive shock).

    Layout modes
    ------------
    Default (both panels, symmetric y-axis):
        Each row shows neg (left) and pos (right) panels with a symmetric y-axis
        ``[-max, +max]`` so both directions are visible simultaneously.

    agg_consumption_mode=True (both panels, one-sided y-axis):
        Intended for the main aggregate-consumption IR figure.  The negative panel
        spans ``[-magnitude, 0]`` and the positive panel spans ``[0, magnitude]``
        where ``magnitude`` is the largest absolute response across either panel.
        This maximises vertical resolution since each panel uses its full height for
        the relevant territory.

    negative_only=True (single panel, symmetric y-axis):
        Only the negative-shock column is created.  Useful for non-consumption
        aggregate IRs and sectoral IRs where a single large shock is sufficient.

    Parameters:
    -----------
    gir_data : dict
        GIR results from JAX analysis (pos_5, neg_5, pos_5_stochss, neg_5_stochss …).
    matlab_ir_data : dict
        Dictionary from load_matlab_irs containing MATLAB IRs.
    sector_idx : int
        Sector index (0-based) to plot.
    sector_label : str
        Label for the sector.
    variable_to_plot : str
        Analysis variable to plot (default: "Agg. Consumption").
    shock_sizes : list
        Shock sizes to show, one row per size (e.g., [5, 10, 20]).
    figsize : tuple, optional
        Figure size.  Defaults to (10, 5*n_sizes) for two-panel modes and
        (7, 4*n_sizes) for negative_only.
    save_dir : str, optional
        Directory to save the figure.
    analysis_name : str, optional
        Name for the analysis (used in the saved file name).
    display_dpi : int
        DPI for display.
    max_periods : int
        Maximum periods to plot.
    n_sectors : int
        Number of sectors in the model.
    benchmark_method : str, optional
        Backward-compatible single MATLAB benchmark overlay.
    benchmark_methods : list[str], optional
        MATLAB benchmarks to overlay ("PerfectForesight", "FirstOrder", "SecondOrder").
    response_source : str
        Which DEQN response to plot ("GIR" or "IR_stoch_ss").
    agg_consumption_mode : bool
        If True, use one-sided y-axis (neg panel: neg territory, pos panel: pos territory).
    negative_only : bool
        If True, create only the negative-shock panel (single column).
    filename_suffix : str, optional
        Extra token appended to the saved filename before ``analysis_name`` so
        alternative layouts can coexist without overwriting one another.

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    from DEQN.econ_models.RbcProdNet_April2026.matlab_irs import (
        get_matlab_ir_fixedprice,
        get_matlab_ir_for_analysis_variable,
    )

    def _should_print_global_consumption_ir() -> bool:
        return variable_to_plot == "Agg. Consumption" and filename_suffix is None

    def _format_ir_values(series: np.ndarray) -> str:
        return np.array2string(
            np.asarray(series, dtype=float),
            separator=", ",
            max_line_width=1_000_000,
            formatter={"float_kind": lambda x: f"{x:.10f}"},
        )

    def _print_global_consumption_ir(*, experiment_name: str, sign_label: str, shock_size_value, series: np.ndarray) -> None:
        if not _should_print_global_consumption_ir():
            return
        print(
            "      Global solution aggregate consumption IR "
            f"| sector={sector_label} "
            f"| response={response_source} "
            f"| experiment={experiment_name} "
            f"| sign={sign_label} "
            f"| shock={shock_size_value}%",
            flush=True,
        )
        print(f"        {_format_ir_values(series)}", flush=True)

    def _format_solution_ir_label(exp_name: str, response_kind: str, distinguish_response_kinds: bool) -> str:
        response_labels = {
            "GIR": "GIR",
            "IR_stoch_ss": "Stochastic SS IR",
        }
        if not distinguish_response_kinds:
            return exp_name
        return f"{exp_name} ({response_labels.get(response_kind, response_kind)})"

    benchmark_key_map = {
        "FirstOrder": "first_order",
        "SecondOrder": "second_order",
        "PerfectForesight": "perfect_foresight",
    }
    benchmark_label_map = {
        "FirstOrder": "First Order",
        "SecondOrder": "Second Order",
        "PerfectForesight": "Perfect Foresight",
    }
    resolved_benchmark_methods = _resolve_ir_benchmark_methods(
        benchmark_methods=benchmark_methods,
        benchmark_method=benchmark_method,
    )
    benchmark_series_specs = [
        (
            benchmark_label_map.get(method, method),
            benchmark_key_map.get(method, "perfect_foresight"),
        )
        for method in resolved_benchmark_methods
    ]

    def _resolve_requested_shock_keys(ir_lookup, sign_prefix: str, requested_size) -> list[str]:
        if not ir_lookup:
            return []

        requested_token = format_shock_size_token(requested_size)
        exact_key = build_shock_key(sign_prefix, requested_size)
        if exact_key in ir_lookup:
            return [exact_key]

        numeric_matches = []
        for key in ir_lookup:
            if not key.startswith(f"{sign_prefix}_"):
                continue
            suffix = key[len(sign_prefix) + 1 :]
            key_value = parse_shock_size_token(suffix)
            if key_value is None:
                continue
            if abs(key_value - float(requested_size)) <= 1e-6:
                numeric_matches.append(key)
        return numeric_matches

    n_sizes = len(shock_sizes)
    n_cols = 1 if negative_only else 2

    if figsize is None:
        if negative_only:
            figsize = (5.2, 4.8 * n_sizes)
        else:
            figsize = (8.8, 3.6 * n_sizes)

    fig, axes = plt.subplots(n_sizes, n_cols, figsize=figsize, dpi=display_dpi, sharex=True, squeeze=False)

    experiment_name = _single_experiment_name(gir_data, "plot_sector_ir_by_shock_size") if gir_data else None
    distinguish_response_kinds = False
    first_exp_data = gir_data[experiment_name] if experiment_name else {}
    all_state_names = list(first_exp_data.keys()) if first_exp_data else []

    state_name = None
    for candidate in all_state_names:
        gir_info = first_exp_data[candidate]
        candidate_state_idx = gir_info.get("state_idx")
        if candidate_state_idx == sector_idx or candidate_state_idx == n_sectors + sector_idx:
            state_name = candidate
            break

    if state_name is None:
        for possible_idx in [sector_idx, n_sectors + sector_idx]:
            possible_name = f"state_{possible_idx}"
            if possible_name in all_state_names:
                state_name = possible_name
                break

    time_periods = np.arange(max_periods)

    for j, shock_size in enumerate(shock_sizes):
        ax_neg = axes[j, 0]
        ax_pos = axes[j, 1] if not negative_only else None

        pos_key = build_shock_key("pos", shock_size)
        neg_key = build_shock_key("neg", shock_size)
        pos_stochss_key = build_shock_key("pos", shock_size, suffix="_stochss")
        neg_stochss_key = build_shock_key("neg", shock_size, suffix="_stochss")

        skip_initial_matlab = variable_to_plot != "Kj"
        if ergodic_price_aggregation and policies_ss is not None and P_ergodic is not None:
            matlab_irs = get_matlab_ir_fixedprice(
                matlab_ir_data,
                sector_idx,
                variable_to_plot,
                policies_ss=policies_ss,
                P_ergodic=P_ergodic,
                n_sectors=n_sectors,
                Pk_ergodic=Pk_ergodic,
                state_ss=state_ss,
                max_periods=max_periods,
                skip_initial=skip_initial_matlab,
            )
        else:
            matlab_irs = get_matlab_ir_for_analysis_variable(
                matlab_ir_data,
                sector_idx,
                variable_to_plot,
                max_periods,
                skip_initial=skip_initial_matlab,
            )
        row_abs_max = 0.0
        row_min = np.inf
        row_max = -np.inf

        def _plot_line(ax, series, *, label=None, color=None, linewidth=1.5, linestyle="-", marker=None, alpha=0.8):
            nonlocal row_abs_max, row_min, row_max
            if series is None:
                return
            arr = np.asarray(series)
            if arr.size == 0:
                return
            t = np.arange(min(len(arr), max_periods))
            y = arr[:max_periods]
            ax.plot(
                t,
                y,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker=marker,
                alpha=alpha,
                label=label,
            )
            finite = np.isfinite(y)
            if np.any(finite):
                row_abs_max = max(row_abs_max, float(np.max(np.abs(y[finite]))))
                row_min = min(row_min, float(np.min(y[finite])))
                row_max = max(row_max, float(np.max(y[finite])))

        if matlab_irs:
            pos_keys = _resolve_requested_shock_keys(matlab_irs, "pos", shock_size)
            neg_keys = _resolve_requested_shock_keys(matlab_irs, "neg", shock_size)
            if not pos_keys and not neg_keys:
                print(
                    f"      Warning: no MATLAB IR benchmark found for requested shock size {shock_size}. "
                    f"Available keys: {sorted(matlab_irs.keys())}"
                )

            for benchmark_rank, (benchmark_label, benchmark_series_key) in enumerate(benchmark_series_specs):
                if not negative_only and ax_pos is not None:
                    for pk in pos_keys:
                        pos_benchmark = matlab_irs[pk].get(benchmark_series_key)
                        if pos_benchmark is not None:
                            pos_benchmark = pos_benchmark[:max_periods] * 100
                            style = _benchmark_style(benchmark_rank)
                            _plot_line(
                                ax_pos,
                                pos_benchmark,
                                color=style["color"],
                                linewidth=style["linewidth"],
                                linestyle=style["linestyle"],
                                marker=style["marker"],
                                alpha=style["alpha"],
                            )

                neg_label_added = False
                for nk in neg_keys:
                    neg_benchmark = matlab_irs[nk].get(benchmark_series_key)
                    if neg_benchmark is not None:
                        neg_benchmark = neg_benchmark[:max_periods] * 100
                        style = _benchmark_style(benchmark_rank)
                        _plot_line(
                            ax_neg,
                            neg_benchmark,
                            color=style["color"],
                            linewidth=style["linewidth"],
                            linestyle=style["linestyle"],
                            marker=style["marker"],
                            alpha=style["alpha"],
                            label=benchmark_label if (j == 0 and not neg_label_added) else None,
                        )
                        neg_label_added = True
        elif j == 0:
            available_sector_indices = sorted(
                {
                    sidx
                    for shock_data in matlab_ir_data.values()
                    for sidx in (shock_data.get("sectors", {}) or {}).keys()
                }
            )
            print(
                f"      Warning: no MATLAB IRs found for sector {sector_idx + 1}, variable '{variable_to_plot}'. "
                f"Available sectors (python 0-based): {available_sector_indices}"
            )

        if state_name and experiment_name and state_name in gir_data[experiment_name]:
            state_gir_data = gir_data[experiment_name][state_name]
            if response_source == "GIR":
                pos_response_key = pos_key
                neg_response_key = neg_key
            else:
                pos_response_key = pos_stochss_key
                neg_response_key = neg_stochss_key

            if not negative_only and ax_pos is not None and pos_response_key in state_gir_data:
                gir_vars_pos = state_gir_data[pos_response_key].get("gir_analysis_variables", {})
                if variable_to_plot in gir_vars_pos:
                    response_pos = gir_vars_pos[variable_to_plot][:max_periods] * 100
                    _print_global_consumption_ir(
                        experiment_name=experiment_name,
                        sign_label="positive",
                        shock_size_value=shock_size,
                        series=response_pos,
                    )
                    style = _experiment_style(0, response_source)
                    _plot_line(
                        ax_pos,
                        response_pos,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        alpha=style["alpha"],
                    )

            if neg_response_key in state_gir_data:
                gir_vars_neg = state_gir_data[neg_response_key].get("gir_analysis_variables", {})
                if variable_to_plot in gir_vars_neg:
                    response_neg = gir_vars_neg[variable_to_plot][:max_periods] * 100
                    _print_global_consumption_ir(
                        experiment_name=experiment_name,
                        sign_label="negative",
                        shock_size_value=shock_size,
                        series=response_neg,
                    )
                    style = _experiment_style(0, response_source)
                    _plot_line(
                        ax_neg,
                        response_neg,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        alpha=style["alpha"],
                        label=(
                            _format_solution_ir_label(experiment_name, response_source, distinguish_response_kinds)
                            if j == 0
                            else None
                        ),
                    )

        if row_abs_max <= 0 or not np.isfinite(row_abs_max):
            row_abs_max = 0.1

        ax_neg.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
        ax_neg.grid(True, alpha=0.3)
        ax_neg.set_box_aspect(1)

        if agg_consumption_mode and not negative_only:
            # One-sided y-axis: derive magnitude from the data on each panel so
            # the neg panel spans [-magnitude, 0] and pos panel spans [0, magnitude].
            # Using the same magnitude across both panels keeps them directly comparable.
            def _panel_abs_max(ax_obj):
                vals = [
                    float(y)
                    for line in ax_obj.get_lines()
                    for y in line.get_ydata()
                    if np.isfinite(y)
                ]
                return max((abs(v) for v in vals), default=0.1)

            neg_abs = _panel_abs_max(ax_neg)
            pos_abs = _panel_abs_max(ax_pos) if ax_pos is not None else 0.0
            magnitude = max(neg_abs, pos_abs, 0.1) * 1.08
            ax_neg.set_ylim(-magnitude, 0)
            if ax_pos is not None:
                ax_pos.set_ylim(0, magnitude)
        elif negative_only:
            if not np.isfinite(row_min) or not np.isfinite(row_max):
                row_min, row_max = -0.1, 0.0
            if row_max <= 0:
                lower = row_min * 1.08 if row_min < 0 else -0.1
                upper = 0.0
            elif row_min >= 0:
                lower = 0.0
                upper = row_max * 1.08 if row_max > 0 else 0.1
            else:
                span = max(row_max - row_min, 0.1)
                pad = 0.08 * span
                lower = row_min - pad
                upper = row_max + pad
            if upper <= lower:
                upper = lower + 0.1
            ax_neg.set_ylim(lower, upper)
        else:
            y_lim_abs = row_abs_max * 1.08
            ax_neg.set_ylim(-y_lim_abs, y_lim_abs)
            if ax_pos is not None:
                ax_pos.set_ylim(-y_lim_abs, y_lim_abs)

        if ax_pos is not None:
            ax_pos.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
            ax_pos.grid(True, alpha=0.3)
            ax_pos.set_box_aspect(1)
            ax_pos.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
            ax_pos.set_xlim(0, max_periods - 1)

        ax_neg.set_ylabel(f"{shock_size}% shock\n(% change)", fontweight="bold", fontsize=MEDIUM_SIZE)
        ax_neg.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)
        ax_neg.set_xlim(0, max_periods - 1)

    axes[-1, 0].set_xlabel("Periods", fontsize=SMALL_SIZE)
    if not negative_only:
        axes[-1, 1].set_xlabel("Periods", fontsize=SMALL_SIZE)

    if not negative_only:
        axes[0, 0].set_title("Negative shock", fontsize=SMALL_SIZE, color="gray")
        axes[0, 1].set_title("Positive shock", fontsize=SMALL_SIZE, color="gray")

    print(f"      IR plot: [{variable_to_plot}]  {sector_label} TFP Shock")

    # Legend: always on the negative panel (col 0, row 0) so it is visible regardless of mode.
    handles_neg, labels_neg = axes[0, 0].get_legend_handles_labels()
    if handles_neg:
        axes[0, 0].legend(
            handles_neg,
            labels_neg,
            loc="upper right",
            fontsize=SMALL_SIZE - 1,
            framealpha=0.9,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_dir:

        def make_safe_filename(label):
            return label.replace(" ", "_").replace(".", "").replace("/", "_")

        safe_sector = make_safe_filename(sector_label)
        safe_var = make_safe_filename(variable_to_plot)
        safe_filename_suffix = make_safe_filename(str(filename_suffix)) if filename_suffix else None
        filename_stem = f"IR_{safe_var}_{safe_sector}"
        if safe_filename_suffix:
            filename_stem = f"{filename_stem}_{safe_filename_suffix}"
        if analysis_name:
            filename = f"{filename_stem}_{analysis_name}.png"
        else:
            filename = f"{filename_stem}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
        _write_figure_note_tex(
            save_path,
            _build_ir_note(
                variable_to_plot=variable_to_plot,
                sector_label=sector_label,
                shock_sizes=shock_sizes,
                benchmark_methods=resolved_benchmark_methods,
                response_source=response_source,
                negative_only=negative_only,
                is_aggregate=variable_to_plot.startswith("Agg.") or variable_to_plot == "Intratemporal Utility",
            ),
        )
        _print_saved_file(save_path, indent="      ")

    return fig, axes


def plot_sectoral_capital_stochss(
    stochastic_ss_states: Dict[str, Any],
    save_dir: str,
    analysis_name: str,
    econ_model: Any,
    figsize: Tuple[float, float] = (12, 8),
    display_dpi: int = 100,
):
    """
    Create publication-quality bar graph of sectoral capital at the stochastic steady state.

    This plot shows the sectoral capital distribution at the stochastic steady state,
    which is where the economy converges when there are no future shocks but starting
    from the ergodic distribution.

    Parameters:
    -----------
    stochastic_ss_states : dict
        Dictionary mapping experiment labels to stochastic SS states (in logdev form).
        Each state array has shape (n_states,) where first n_sectors entries are capital.
    save_dir : str
        Directory where the figure should be saved
    analysis_name : str
        Name of the analysis to include in the filename
    econ_model : Any
        Economic model instance (used to get n_sectors and labels)
    figsize : tuple, optional
        Figure size (width, height) in inches
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels
    experiment_name = _single_experiment_name(stochastic_ss_states, "plot_sectoral_capital_stochss")

    stoch_ss_state = stochastic_ss_states[experiment_name]
    sectoral_capital = stoch_ss_state[:n_sectors]

    sorted_indices = np.argsort(sectoral_capital)[::-1]
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    x = np.arange(n_sectors)
    bar_width = 0.8

    sorted_capital = sectoral_capital[sorted_indices]
    ax.bar(
        x,
        sorted_capital * 100,
        bar_width,
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Stochastic SS Capital (% Dev. from Deterministic SS)", fontweight="bold", fontsize=MEDIUM_SIZE)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"sectoral_capital_stochss_{analysis_name}.png" if analysis_name else "sectoral_capital_stochss.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    _print_saved_file(save_path)
    plt.show()

    return fig, ax


def plot_sectoral_capital_comparison(
    simul_obs: Any,
    stochastic_ss_state: Any,
    save_dir: str,
    analysis_name: str,
    econ_model: Any,
    experiment_label: str,
    figsize: Tuple[float, float] = (14, 8),
    display_dpi: int = 100,
):
    """
    Create publication-quality bar graph comparing ergodic mean vs stochastic SS sectoral capital.

    This plot shows both the average sectoral capital from the ergodic distribution
    and the sectoral capital at the stochastic steady state side by side.

    Parameters:
    -----------
    simul_obs : array
        Simulation observations array of shape (n_periods, n_obs)
    stochastic_ss_state : array
        Stochastic steady state (in logdev form) with shape (n_states,)
    save_dir : str
        Directory where the figure should be saved
    analysis_name : str
        Name of the analysis to include in the filename
    econ_model : Any
        Economic model instance (used to get n_sectors and labels)
    experiment_label : str
        Label for the experiment
    figsize : tuple, optional
        Figure size (width, height) in inches
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels

    # Get ergodic mean capital
    ergodic_capital = np.mean(simul_obs, axis=0)[:n_sectors]

    # Get stochastic SS capital
    stochss_capital = stochastic_ss_state[:n_sectors]

    # Sort by ergodic capital values
    sorted_indices = np.argsort(ergodic_capital)[::-1]
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]
    sorted_ergodic = ergodic_capital[sorted_indices]
    sorted_stochss = stochss_capital[sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    x = np.arange(n_sectors)
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2,
        sorted_ergodic * 100,
        bar_width,
        label="Ergodic Mean",
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.bar(
        x + bar_width / 2,
        sorted_stochss * 100,
        bar_width,
        label="Stochastic SS",
        color=colors[1],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Capital (% Deviations from Deterministic SS)", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_title(
        f"Sectoral Capital: Ergodic Mean vs Stochastic SS ({experiment_label})", fontweight="bold", fontsize=LARGE_SIZE
    )

    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=SMALL_SIZE)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_label = experiment_label.replace(" ", "_").replace(".", "").replace("/", "_")
    filename = (
        f"sectoral_capital_comparison_{safe_label}_{analysis_name}.png"
        if analysis_name
        else f"sectoral_capital_comparison_{safe_label}.png"
    )
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    _print_saved_file(save_path)
    plt.show()

    return fig, ax


def plot_sectoral_variable_stochss(
    stochastic_ss_states: Dict[str, Any],
    stochastic_ss_policies: Dict[str, Any],
    variable_name: str,
    save_dir: str,
    analysis_name: str,
    econ_model: Any,
    upstreamness_data: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (12, 8),
    display_dpi: int = 100,
):
    """
    Create publication-quality bar graph of a sectoral variable at the stochastic steady state.

    This plot shows the sectoral distribution of a variable at the stochastic steady state,
    which is where the economy converges when there are no future shocks but starting
    from the ergodic distribution.

    Parameters:
    -----------
    stochastic_ss_states : dict
        Dictionary mapping experiment labels to stochastic SS states (in logdev form).
        Each state array has shape (n_states,) where first n_sectors entries are capital.
    stochastic_ss_policies : dict
        Dictionary mapping experiment labels to stochastic SS policies (in logdev form).
    variable_name : str
        Name of the variable to plot. Options: "K" (capital), "L" (labor), "Y" (value added),
        "M" (intermediates), "Q" (gross output)
    save_dir : str
        Directory where the figure should be saved
    analysis_name : str
        Name of the analysis to include in the filename
    econ_model : Any
        Economic model instance (used to get n_sectors, labels, and policies_ss)
    upstreamness_data : dict, optional
        Dictionary containing upstreamness measures with keys "U_M" (IO upstreamness)
        and "U_I" (investment upstreamness). If provided, correlations are displayed.
    figsize : tuple, optional
        Figure size (width, height) in inches
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    from scipy import stats

    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels
    experiment_name = _single_experiment_name(stochastic_ss_policies, "plot_sectoral_variable_stochss")

    # Variable name mapping
    variable_info = {
        "K": {"title": "Capital", "index_start": 0, "source": "state"},
        "L": {"title": "Labor", "index_start": n_sectors, "source": "policy"},
        "Y": {"title": "Value Added", "index_start": 10 * n_sectors, "source": "policy"},
        "M": {"title": "Intermediates", "index_start": 4 * n_sectors, "source": "policy"},
        "Q": {"title": "Gross Output", "index_start": 9 * n_sectors, "source": "policy"},
    }

    if variable_name not in variable_info:
        raise ValueError(f"Unknown variable: {variable_name}. Options: {list(variable_info.keys())}")

    var_info = variable_info[variable_name]
    idx_start = var_info["index_start"]
    idx_end = idx_start + n_sectors

    if var_info["source"] == "state":
        data = stochastic_ss_states[experiment_name]
    else:
        data = stochastic_ss_policies[experiment_name]
    sectoral_values = data[idx_start:idx_end]

    sorted_indices = np.argsort(sectoral_values)[::-1]
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    x = np.arange(n_sectors)
    bar_width = 0.8

    sorted_values = sectoral_values[sorted_indices]
    ax.bar(
        x,
        sorted_values * 100,
        bar_width,
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel(
        f"Stochastic SS {var_info['title']} (% Dev. from Deterministic SS)", fontweight="bold", fontsize=MEDIUM_SIZE
    )

    # Calculate and display upstreamness correlations if data provided
    if upstreamness_data is not None:
        U_M = np.array(upstreamness_data["U_M"])
        U_I = np.array(upstreamness_data["U_I"])

        values = np.array(sectoral_values)
        corr_M, p_M = stats.pearsonr(values, U_M)
        corr_I, p_I = stats.pearsonr(values, U_I)
        sig_M = "***" if p_M < 0.01 else "**" if p_M < 0.05 else "*" if p_M < 0.1 else ""
        sig_I = "***" if p_I < 0.01 else "**" if p_I < 0.05 else "*" if p_I < 0.1 else ""
        corr_text = f"ρ(IO Upstr.)={corr_M:.2f}{sig_M}, ρ(Inv Upstr.)={corr_I:.2f}{sig_I}"
        ax.text(
            0.98,
            0.98,
            corr_text,
            transform=ax.transAxes,
            fontsize=SMALL_SIZE + 1,
            verticalalignment="top",
            horizontalalignment="right",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.88, edgecolor="gray"),
        )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_var_name = variable_name.lower()
    filename = (
        f"sectoral_{safe_var_name}_stochss_{analysis_name}.png"
        if analysis_name
        else f"sectoral_{safe_var_name}_stochss.png"
    )
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    _write_figure_note_tex(
        save_path,
        _build_sectoral_distribution_note(
            variable_title=var_info["title"],
            display_labels=[experiment_name],
            source_kind="stochss",
            include_upstreamness=upstreamness_data is not None,
        ),
    )
    _print_saved_file(save_path)
    plt.show()

    return fig, ax


def plot_sectoral_variable_ergodic(
    raw_simulation_data: Dict[str, Any],
    variable_name: str,
    save_dir: str,
    analysis_name: str,
    econ_model: Any,
    upstreamness_data: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (12, 8),
    display_dpi: int = 100,
):
    """
    Create publication-quality bar graph of a sectoral variable from the ergodic distribution.

    This plot shows the sectoral distribution of the ergodic mean (time-average from simulation)
    of a variable for the single analyzed experiment.

    Parameters:
    -----------
    raw_simulation_data : dict
        Dictionary mapping experiment labels to simulation data dictionaries.
        Each contains "simul_obs" and "simul_policies" arrays.
    variable_name : str
        Name of the variable to plot. Options: "K" (capital), "L" (labor), "Y" (value added),
        "M" (intermediates), "Q" (gross output)
    save_dir : str
        Directory where the figure should be saved
    analysis_name : str
        Name of the analysis to include in the filename
    econ_model : Any
        Economic model instance (used to get n_sectors, labels, and policies_ss)
    upstreamness_data : dict, optional
        Dictionary containing upstreamness measures with keys "U_M" (IO upstreamness)
        and "U_I" (investment upstreamness). If provided, correlations are displayed.
    figsize : tuple, optional
        Figure size (width, height) in inches
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    from scipy import stats

    n_sectors = econ_model.n_sectors
    sector_labels = econ_model.labels
    experiment_name = _single_experiment_name(raw_simulation_data, "plot_sectoral_variable_ergodic")

    # Variable name mapping
    variable_info = {
        "K": {"title": "Capital", "index_start": 0, "source": "state"},
        "L": {"title": "Labor", "index_start": n_sectors, "source": "policy"},
        "Y": {"title": "Value Added", "index_start": 10 * n_sectors, "source": "policy"},
        "M": {"title": "Intermediates", "index_start": 4 * n_sectors, "source": "policy"},
        "Q": {"title": "Gross Output", "index_start": 9 * n_sectors, "source": "policy"},
    }

    if variable_name not in variable_info:
        raise ValueError(f"Unknown variable: {variable_name}. Options: {list(variable_info.keys())}")

    var_info = variable_info[variable_name]
    idx_start = var_info["index_start"]
    idx_end = idx_start + n_sectors

    if var_info["source"] == "state":
        data = raw_simulation_data[experiment_name]["simul_obs"]
    else:
        data = raw_simulation_data[experiment_name]["simul_policies"]
    sectoral_values = np.mean(data[:, idx_start:idx_end], axis=0)

    sorted_indices = np.argsort(sectoral_values)[::-1]
    sorted_sector_labels = [sector_labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    x = np.arange(n_sectors)
    bar_width = 0.8

    sorted_values = sectoral_values[sorted_indices]
    ax.bar(
        x,
        sorted_values * 100,
        bar_width,
        color=colors[0],
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sector_labels, rotation=45, ha="right")

    ax.tick_params(axis="both", which="major")

    ax.set_xlabel("Sector", fontweight="bold", fontsize=MEDIUM_SIZE)
    ax.set_ylabel(
        f"Ergodic Mean {var_info['title']} (% Dev. from Deterministic SS)", fontweight="bold", fontsize=MEDIUM_SIZE
    )

    # Calculate and display upstreamness correlations if data provided
    if upstreamness_data is not None:
        U_M = np.array(upstreamness_data["U_M"])
        U_I = np.array(upstreamness_data["U_I"])

        values = np.array(sectoral_values)
        corr_M, p_M = stats.pearsonr(values, U_M)
        corr_I, p_I = stats.pearsonr(values, U_I)
        sig_M = "***" if p_M < 0.01 else "**" if p_M < 0.05 else "*" if p_M < 0.1 else ""
        sig_I = "***" if p_I < 0.01 else "**" if p_I < 0.05 else "*" if p_I < 0.1 else ""
        corr_text = f"ρ(IO Upstr.)={corr_M:.2f}{sig_M}, ρ(Inv Upstr.)={corr_I:.2f}{sig_I}"
        ax.text(
            0.98,
            0.98,
            corr_text,
            transform=ax.transAxes,
            fontsize=SMALL_SIZE + 1,
            verticalalignment="top",
            horizontalalignment="right",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.88, edgecolor="gray"),
        )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_var_name = variable_name.lower()
    filename = (
        f"sectoral_{safe_var_name}_ergodic_{analysis_name}.png"
        if analysis_name
        else f"sectoral_{safe_var_name}_ergodic.png"
    )
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    _write_figure_note_tex(
        save_path,
        _build_sectoral_distribution_note(
            variable_title=var_info["title"],
            display_labels=[experiment_name],
            source_kind="ergodic",
            include_upstreamness=upstreamness_data is not None,
        ),
    )
    _print_saved_file(save_path)
    plt.show()

    return fig, ax


def plot_gir_heatmap(
    gir_data: Dict[str, Any],
    aggregate_idx: int = 0,
    aggregate_labels: list = [
        "Agg. Consumption",
        "Agg. Investment",
        "Agg. GDP",
        "Agg. Capital",
        "Agg. Labor",
        "Intratemporal Utility",
    ],
    time_slice: int = 10,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
    analysis_name: Optional[str] = None,
    display_dpi: int = 100,
):
    """
    Create a heatmap showing GIR responses across sectors at a specific time period.

    Parameters:
    -----------
    gir_data : dict
        Dictionary containing GIR results from analysis
    aggregate_idx : int, optional
        Which aggregate variable to plot (index 0-6). Default is consumption.
    aggregate_labels : list, optional
        Labels for the aggregate variables
    time_slice : int, optional
        Which time period to show in the heatmap
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path
    analysis_name : str, optional
        Name of the analysis to include in the filename
    display_dpi : int, optional
        DPI for display (default 100). Saved figures always use 300 DPI.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    first_experiment = _single_experiment_name(gir_data, "plot_gir_heatmap")
    exp_data = gir_data[first_experiment]

    # Get all sector names and their responses
    sector_names = list(exp_data.keys())
    n_sectors = len(sector_names)

    # Extract responses at the specified time slice
    responses = []
    sector_labels = []

    for sector_name in sector_names:
        gir_aggregates = exp_data[sector_name]["gir_aggregates"]
        # Convert to percentage and get specific aggregate at time slice
        response_pct = gir_aggregates[time_slice, aggregate_idx] * 100
        responses.append(response_pct)
        sector_labels.append(sector_name)

    # Create matrix for heatmap (single row)
    response_matrix = np.array(responses).reshape(1, -1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=display_dpi)

    # Create heatmap
    im = ax.imshow(response_matrix, cmap="RdBu_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(n_sectors))
    ax.set_xticklabels(sector_labels, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels([f"Period {time_slice}"])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{aggregate_labels[aggregate_idx]} (% change)", fontweight="bold", fontsize=MEDIUM_SIZE)

    # Apply consistent styling
    ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        # Modify filename to include analysis name if provided
        var_names = ["C_agg", "I_agg", "GDP_agg", "K_agg", "L_agg", "utility_intratemp"]
        if analysis_name:
            base_filename = os.path.basename(save_path)
            ext = os.path.splitext(base_filename)[1] or ".png"
            new_filename = f"GIR_heatmap_{var_names[aggregate_idx]}_period{time_slice}_{analysis_name}{ext}"
            final_save_path = save_path.replace(os.path.basename(save_path), new_filename)
        else:
            final_save_path = save_path.replace(
                os.path.basename(save_path), f"GIR_heatmap_{var_names[aggregate_idx]}_period{time_slice}.png"
            )
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight", format="png")
        _print_saved_file(final_save_path)

    return fig, ax
