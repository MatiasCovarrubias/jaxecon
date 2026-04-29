import os


DEFAULT_IR_BENCHMARK_METHODS = ["PerfectForesight", "FirstOrder"]


def analysis_named_path(directory, stem, analysis_name, extension):
    suffix = f"_{analysis_name}" if analysis_name else ""
    return os.path.join(directory, f"{stem}{suffix}{extension}")


def escape_latex(text):
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


def make_safe_plot_label(label):
    return label.replace(" ", "_").replace(".", "").replace("/", "_")


def latex_relative_path(path, base_dir):
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def tex_fragment_has_table_env(tex_path):
    if not os.path.exists(tex_path):
        return False
    with open(tex_path) as tex_file:
        return r"\begin{table}" in tex_file.read()


def figure_note_path(figure_path):
    return os.path.splitext(figure_path)[0] + "_note.tex"


def join_labels(labels):
    labels = [label for label in labels if label]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def build_simple_figure_spec(path, caption, note_text=None, note_path=None):
    return {
        "path": path,
        "caption": caption,
        "note_text": note_text,
        "note_path": note_path,
    }


def caption_label(label):
    return label if str(label).isupper() else str(label).lower()


def format_percent_list(values):
    def format_percent_value(value):
        rounded = round(float(value), 8)
        if float(rounded).is_integer():
            return str(int(round(rounded)))
        return f"{rounded:.8f}".rstrip("0").rstrip(".")

    values = [format_percent_value(value) for value in values if value is not None]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def resolve_ir_benchmark_methods(config_dict):
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


def describe_ir_benchmark_methods(config_dict):
    benchmark_method_labels = {
        "FirstOrder": "1st-order approximation",
        "SecondOrder": "2nd-order approximation",
        "PerfectForesight": "perfect foresight",
        "MITShocks": "MIT shocks",
    }
    labels = [
        benchmark_method_labels.get(method, method)
        for method in resolve_ir_benchmark_methods(config_dict)
    ]
    return join_labels(labels)


def resolve_ir_response_source(config_dict):
    use_gir = config_dict.get("use_gir")
    if use_gir is not None:
        return "GIR" if bool(use_gir) else "IR_stoch_ss"

    configured_methods = config_dict.get("ir_methods")
    if configured_methods is None:
        legacy_method = config_dict.get("ir_method")
        configured_methods = [legacy_method] if legacy_method else []
    elif isinstance(configured_methods, str):
        configured_methods = [configured_methods]

    return "GIR" if "GIR" in configured_methods else "IR_stoch_ss"


def describe_deqn_ir_note(config_dict):
    if resolve_ir_response_source(config_dict) == "GIR":
        return (
            "Solid lines report the DEQN generalized impulse response: initial states are sampled from the "
            "ergodic distribution and, for each draw, a zero-shock baseline path is compared with a path "
            "whose TFP state is shifted on impact, with both paths then simulated forward with zero shocks; "
            "the plotted response is the average difference across draws. "
        )

    return (
        "Solid lines report the DEQN stochastic-steady-state impulse response: the baseline and "
        "counterfactual paths start from the stochastic steady state, the TFP shock is applied on impact, "
        "and both paths are then simulated forward with zero shocks. "
    )


def existing_subfigures(figure_specs):
    return [figure_spec for figure_spec in figure_specs if os.path.exists(figure_spec["path"])]


def existing_subfigure_groups(group_specs):
    existing_groups = []
    for group_spec in group_specs:
        subfigures = existing_subfigures(group_spec.get("subfigures", []))
        if subfigures:
            existing_group = dict(group_spec)
            existing_group["subfigures"] = subfigures
            existing_groups.append(existing_group)
    return existing_groups
