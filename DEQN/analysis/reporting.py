import os

from DEQN.analysis.report_specs import (
    analysis_named_path,
    build_simple_figure_spec,
    caption_label,
    escape_latex,
    describe_deqn_ir_note,
    describe_ir_benchmark_methods,
    existing_subfigure_groups,
    existing_subfigures,
    figure_note_path,
    format_percent_list,
    join_labels,
    latex_relative_path,
    make_safe_plot_label,
    tex_fragment_has_table_env,
)

analysis_hooks = None


def _build_output_note_context(config_dict, matlab_common_shock_schedule=None):
    periods_per_episode = int(config_dict.get("periods_per_epis", 0) or 0)
    burn_in_periods = int(config_dict.get("burn_in_periods", 0) or 0)
    n_simul_seeds = int(config_dict.get("n_simul_seeds", 0) or 0)
    retained_periods_per_seed = max(periods_per_episode - burn_in_periods, 0)
    note_context = {
        "long_simulation": bool(config_dict.get("long_simulation", False)),
        "periods_per_episode": periods_per_episode,
        "burn_in_periods": burn_in_periods,
        "n_simul_seeds": n_simul_seeds,
        "retained_periods_per_seed": retained_periods_per_seed,
        "total_retained_periods": retained_periods_per_seed * n_simul_seeds,
        "common_shock_burn_in": None,
        "common_shock_active_periods": None,
        "common_shock_burn_out": None,
        "common_shock_total_periods": None,
    }

    schedule = matlab_common_shock_schedule or {}
    active_shocks = schedule.get("active_shocks")
    full_shocks = schedule.get("full_shocks")
    note_context.update(
        {
            "common_shock_burn_in": int(schedule.get("burn_in", 0)),
            "common_shock_active_periods": int(active_shocks.shape[0]) if active_shocks is not None else None,
            "common_shock_burn_out": int(schedule.get("burn_out", 0)),
            "common_shock_total_periods": int(full_shocks.shape[0]) if full_shocks is not None else None,
        }
    )
    return note_context


def _build_analysis_latex_sections(*, config_dict, analysis_dir, simulation_dir, irs_dir, econ_model):
    if analysis_hooks is not None and hasattr(analysis_hooks, "get_report_sections"):
        hook_sections = analysis_hooks.get_report_sections(
            config=config_dict,
            analysis_dir=analysis_dir,
            simulation_dir=simulation_dir,
            irs_dir=irs_dir,
            econ_model=econ_model,
            helpers={
                "analysis_named_path": analysis_named_path,
                "build_simple_figure_spec": build_simple_figure_spec,
                "make_safe_plot_label": make_safe_plot_label,
                "caption_label": caption_label,
                "join_labels": join_labels,
                "format_percent_list": format_percent_list,
                "describe_ir_benchmark_methods": describe_ir_benchmark_methods,
                "describe_deqn_ir_note": describe_deqn_ir_note,
                "existing_subfigures": existing_subfigures,
                "existing_subfigure_groups": existing_subfigure_groups,
            },
        )
        if hook_sections is not None:
            return hook_sections

    analysis_name = config_dict.get("analysis_name") or "analysis"
    section_specs = [
        ("Model vs. Data Moments", [os.path.join(analysis_dir, f"calibration_table_{analysis_name}.tex")]),
        ("Aggregate Stochastic Steady State", [os.path.join(analysis_dir, f"stochastic_ss_aggregates_{analysis_name}.tex")]),
        ("Descriptive Statistics", [os.path.join(simulation_dir, f"descriptive_stats_{analysis_name}.tex")]),
        ("Welfare Cost of Business Cycles", [os.path.join(analysis_dir, f"welfare_{analysis_name}.tex")]),
    ]
    sections = []
    for title, paths in section_specs:
        existing_paths = [path for path in paths if os.path.exists(path)]
        if existing_paths:
            sections.append({"title": title, "tables": existing_paths, "figures": []})
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
        r"\newlength{\FigHtriple}",
        r"\setlength{\FigHfirst}{0.70\textheight}",
        r"\setlength{\FigHsingle}{0.58\textheight}",
        r"\setlength{\FigHdouble}{0.40\textheight}",
        r"\setlength{\FigHpanel}{0.23\textheight}",
        r"\setlength{\FigHtriple}{0.18\textheight}",
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

    def _grouped_subfigure_layout(subfigure_count):
        if subfigure_count >= 3:
            return "0.31\\textwidth", r"\FigHtriple", r"\hfill"
        if subfigure_count == 2:
            return "0.48\\textwidth", r"\FigHdouble", r"\hfill"
        return "0.88\\textwidth", r"\FigHsingle", None

    for section in sections:
        for tex_path in section["tables"]:
            relative_path = latex_relative_path(tex_path, analysis_dir)
            if tex_fragment_has_table_env(tex_path):
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
            has_subfigures = isinstance(figure_path, dict) and "subfigures" in figure_path
            has_subfigure_groups = isinstance(figure_path, dict) and "subfigure_groups" in figure_path
            if isinstance(figure_path, str) or (not has_subfigures and not has_subfigure_groups):
                single_figure = build_simple_figure_spec(figure_path, "") if isinstance(figure_path, str) else figure_path
                is_first_figure = rendered_figure_count == 0
                relative_path = latex_relative_path(single_figure["path"], analysis_dir)
                note_path = single_figure.get("note_path") or figure_note_path(single_figure["path"])
                lines.extend(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[{_single_figure_include_options(is_first_figure=is_first_figure)}]{{{relative_path}}}",
                    ]
                )
                if single_figure.get("caption"):
                    lines.append(rf"\caption{{{escape_latex(single_figure['caption'])}}}")
                if os.path.exists(note_path):
                    lines.extend([r"\par\smallskip", rf"\input{{{latex_relative_path(note_path, analysis_dir)}}}"])
                elif single_figure.get("note_text"):
                    lines.extend(
                        [
                            r"\par\smallskip",
                            r"\begin{minipage}{0.92\textwidth}",
                            r"\footnotesize",
                            rf"\textit{{Notes:}} {escape_latex(single_figure['note_text'])}",
                            r"\end{minipage}",
                        ]
                    )
                lines.extend([r"\end{figure}", r"\clearpage"])
                rendered_figure_count += 1
                continue

            if has_subfigure_groups:
                subfigure_groups = figure_path.get("subfigure_groups", [])
                if not subfigure_groups:
                    continue

                lines.extend([r"\begin{figure}[H]", r"\centering"])
                for group_idx, subfigure_group in enumerate(subfigure_groups):
                    group_title = subfigure_group.get("title")
                    if group_title:
                        lines.extend(
                            [
                                rf"{{\small\textbf{{{escape_latex(group_title)}}}\par}}",
                                r"\smallskip",
                            ]
                        )

                    grouped_subfigures = subfigure_group.get("subfigures", [])
                    width, height_name, column_separator = _grouped_subfigure_layout(len(grouped_subfigures))
                    for idx, subfigure in enumerate(grouped_subfigures):
                        lines.extend(
                            [
                                rf"\begin{{subfigure}}[t]{{{width}}}",
                                r"\centering",
                                rf"\includegraphics[width=\linewidth,height={height_name},keepaspectratio]{{{latex_relative_path(subfigure['path'], analysis_dir)}}}",
                                rf"\caption{{{escape_latex(subfigure.get('caption', ''))}}}",
                                r"\end{subfigure}",
                            ]
                        )
                        if idx != len(grouped_subfigures) - 1:
                            lines.append(column_separator or r"\hfill")

                    if group_idx != len(subfigure_groups) - 1:
                        lines.extend([r"\par\medskip", r"\medskip"])

                lines.append(rf"\caption{{{escape_latex(figure_path.get('caption', ''))}}}")
                note_path = figure_path.get("note_path")
                note_text = figure_path.get("note_text")
                if note_path and os.path.exists(note_path):
                    lines.extend([r"\par\smallskip", rf"\input{{{latex_relative_path(note_path, analysis_dir)}}}"])
                elif note_text:
                    lines.extend(
                        [
                            r"\par\smallskip",
                            r"\begin{minipage}{0.92\textwidth}",
                            r"\footnotesize",
                            rf"\textit{{Notes:}} {escape_latex(note_text)}",
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
                        rf"\includegraphics[width=\linewidth,height={height_name},keepaspectratio]{{{latex_relative_path(subfigure['path'], analysis_dir)}}}",
                        rf"\caption{{{escape_latex(subfigure.get('caption', ''))}}}",
                        r"\end{subfigure}",
                    ]
                )
                if len(subfigures) > 1 and idx % 2 == 0 and idx != len(subfigures) - 1:
                    lines.append(column_separator or r"\hfill")
                elif idx != len(subfigures) - 1:
                    lines.append(r"\par\medskip")

            lines.append(rf"\caption{{{escape_latex(figure_path.get('caption', ''))}}}")
            note_path = figure_path.get("note_path")
            note_text = figure_path.get("note_text")
            if note_path and os.path.exists(note_path):
                lines.extend([r"\par\smallskip", rf"\input{{{latex_relative_path(note_path, analysis_dir)}}}"])
            elif note_text:
                lines.extend(
                    [
                        r"\par\smallskip",
                        r"\begin{minipage}{0.92\textwidth}",
                        r"\footnotesize",
                        rf"\textit{{Notes:}} {escape_latex(note_text)}",
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
