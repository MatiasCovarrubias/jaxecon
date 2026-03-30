# Python Analysis Guide

This document describes the Python-side analysis layer for `RbcProdNet_April2026`.

It complements the MATLAB documentation:

- `MATLAB/CODEBASE_README.md` explains the MATLAB runtime pipeline that builds the saved objects.
- `MATLAB/ModelData_README.md` explains the structure of `ModelData.mat`, `ModelData_simulation.mat`, and `ModelData_IRs.mat`.

This file is about what Python does _after_ those objects exist.

## Scope

The Python analysis layer has one main job:

1. Load MATLAB benchmark objects and trained DEQN checkpoints.
2. Run the nonlinear DEQN simulation exercises.
3. Build analysis variables under the configured aggregation scheme.
4. Produce the final tables and figures shown in the analysis output folders.

Canonical entry point:

- `DEQN/analysis.py`

Model-specific extension points:

- `analysis_hooks.py`
- `aggregation.py`
- `plot_helpers.py`

## Relationship with MATLAB

The Python analysis code depends on MATLAB outputs in three different ways:

1. `ModelData.mat`
   Used for steady state objects, state-space matrices, empirical targets, and benchmark summary statistics.
2. `ModelData_simulation.mat`
   Used for simulation-based benchmark series such as `FirstOrder`, `SecondOrder`, `PerfectForesight`, and `MITShocks`, when those blocks are present.
3. `ModelData_IRs.mat`
   Used as the benchmark source for aggregate and sectoral IR figures.

The key boundary is:

- MATLAB produces benchmark objects and simulation artifacts.
- Python loads them, aligns them with the DEQN nonlinear simulation, and renders the final comparative analysis.

## High-level flow

The main execution flow in `DEQN/analysis.py` is:

1. Read config and resolve the three MATLAB files.
2. Load `ModelData.mat`.
3. Build the Python `Model`.
4. Optionally load `ModelData_simulation.mat` and extract benchmark simulation windows.
5. Build the nonlinear DEQN simulation runner selected by `config["long_simulation"]`.
6. Use that selected nonlinear run as the single DEQN source for stochastic SS, GIRs, welfare, and nonlinear reporting.
7. For each analyzed experiment:

- run the selected nonlinear simulation
- compute stochastic SS from that simulation
- compute GIR objects
- compute welfare inputs

8. Call `analysis_hooks.prepare_postprocess_analysis(...)`.
9. Render tables and figures in the analysis order.

## Rolling change log

Keep only the latest three entries here. Add newest first. Keep each entry to one short bullet focused on the behavioral change, not the implementation details.

- Aggregate histograms are now active again for the six reported aggregates, comparing `Global solution`, `1st Order Approx.`, and `MIT shocks`, and the combined LaTeX wrapper places them immediately after descriptive statistics.
- The nonlinear DEQN analysis now enforces a single simulation source via `long_simulation`: `False` uses the common-shock run for GIRs, stochastic steady state, and ergodic-price aggregation, while `True` uses the long ergodic run.
- IR selection is now controlled by the boolean `use_gir`, aggregate stochastic-SS tables default to all available stochastic-SS methods when no filter is configured, and saved tables / figures print their filenames to the Python console.

## Current defaults and compatibility

The active April 2026 analysis contract is intentionally comprehensive by default.

- All six reported aggregates are always used in aggregate tables, aggregate IR figures, and aggregate histogram figures: `Agg. Consumption`, `Agg. Investment`, `Agg. GDP`, `Agg. Capital`, `Agg. Labor`, and `Intratemporal Utility`.
- The aggregate IR wrapper expects a full panel for each reported aggregate: one row per discovered shock size, with negative shocks in the left column and positive shocks in the right column.
- Shock sizes are no longer meant to be hardcoded in the main config. Python discovers them from `ModelData_IRs.mat` and stores the discovered list back into `config["ir_shock_sizes"]` for downstream rendering and LaTeX assembly.
- IR selection is now controlled by `config["use_gir"]`: `False` renders the stochastic-steady-state IR and `True` renders the generalized impulse response averaged over ergodic draws.
- `config["long_simulation"]` selects the single nonlinear DEQN simulation source used throughout the analysis: `False` uses the common-shock run and `True` uses the long ergodic run.
- The default IR benchmark overlays are `["PerfectForesight", "FirstOrder"]`. The config still accepts `ir_benchmark_methods` to override the set or ordering, and still accepts the legacy single-string `ir_benchmark_method` for backward compatibility.
- Descriptive-statistics and stochastic-steady-state tables include all available simulation methods by default. For aggregate stochastic-SS tables, when `stochss_methods_to_include` is absent or empty, Python now falls back directly to the available keys in `stochastic_ss_data`. Older keys such as `ergodic_methods_to_include`, `stochss_methods_to_include`, `model_vs_data_methods_to_include`, and `descriptive_stats_variables` now act as compatibility filters rather than the intended default workflow.
- Python recomputes Dynare simulation moments through the common aggregation path even when `ergodic_price_aggregation = false`, so MATLAB and Python moments can be compared on identical definitions.
- Saved PNGs and generated table fragments print `Saved: <filename>` to the Python console so Colab output identifies the object that was just written.

## Moment-comparison contract

For the current April pipeline, the intended MATLAB/Python comparison convention is:

- `ergodic_price_aggregation = false` means aggregate `C`, `I`, `GDP`, `L`, `K`, and `Intratemporal Utility` are read directly from the model-implied aggregate endogenous variables.
- Sectoral value added is still treated with fixed prices in both MATLAB and Python: `VA_j = \bar P_j (Q_j - M_j^{out})`. It does not switch to time-varying prices when aggregate re-aggregation is off.
- Volatility calculations are matched to MATLAB's default `std` normalization, so Python uses the sample standard deviation (`N-1`) rather than NumPy's population default.

## Main Python files

### `DEQN/analysis.py`

This is the orchestrator.

It owns:

- config
- file resolution
- MATLAB loading
- experiment loop
- nonlinear simulation dispatch
- stochastic SS / GIR / welfare setup
- final table and figure ordering

It should stay thin. Model-specific logic should live in hooks or in the model folder.

### `DEQN/analysis/simul_analysis.py`

This file provides the nonlinear simulation machinery.

Main responsibilities:

- long ergodic simulation
- common-shock simulation along a fixed shock path
- conversion from simulated states and policies into `analysis_variables_data`

Important distinction:

- `long_simulation = true` makes the long ergodic simulation the main nonlinear analysis object
- `long_simulation = false` makes the common-shock simulation the main nonlinear analysis object

### `DEQN/analysis/stochastic_ss.py`

This computes stochastic steady states from simulated paths and evaluates equilibrium accuracy.

For this model, the stochastic-SS objects are computed from the selected nonlinear simulation source.

### `DEQN/analysis/GIR.py`

This computes GIR or stochastic-SS impulse responses depending on `config["use_gir"]`.

For this model, the IR exercises are anchored to the selected nonlinear simulation source.

### `DEQN/analysis/tables.py`

This file turns already-computed data into LaTeX and console tables.

It does not decide the economics. It formats:

- model vs data moments
- descriptive statistics
- comparative descriptive statistics
- stochastic SS tables
- ergodic aggregate tables
- welfare tables

### `analysis_hooks.py`

This is the model-specific bridge between generic DEQN analysis utilities and `RbcProdNet_April2026`.

Main responsibilities:

- prepare model-specific aggregation context
- choose between model-implied aggregate rows and optional ergodic-price reaggregation
- load simulation-based MATLAB benchmark series into `analysis_variables_data`
- build model-vs-data moment columns
- prepare the IR rendering context
- render model-specific figures

This file is the most important place to read when the displayed analysis looks economically wrong.

### `aggregation.py`

This is where consistent aggregation logic lives.

It is the core of the comparison between:

- DEQN nonlinear simulations
- MATLAB benchmark simulations
- model-vs-data moments

If aggregate moments differ unexpectedly across methods, this file is one of the first places to inspect.

Current contract:

- Default mode: `ergodic_price_aggregation = false`
  Python reads `c_agg`, `l_agg`, `gdp_agg`, `i_agg`, `k_agg`, and `utility_intratemp` directly from the policy arrays.
- Optional mode: `ergodic_price_aggregation = true`
  Python re-aggregates nonlinear simulations, Dynare simulations, and MATLAB IRs using fixed ergodic-mean prices.

### `plot_helpers.py`

This contains model-specific plotting helpers used by the hooks.

Currently used helpers include:

- aggregate IR figures
- sectoral IR figures
- sectoral stochastic-SS bars
- sectoral ergodic-mean bars

It also contains older or currently unused plotting helpers.

## Where to edit presentation

If you want to change the formatting of displayed tables, figures, or the combined LaTeX wrapper, use this map:

- Edit `DEQN/analysis/tables.py` to change table layout, captions, notes, column labels, widths, and the LaTeX environments used by the generated tables.
- Edit `plot_helpers.py` to change the appearance of PNG figures: subplot layout, legends, axis labels, annotation boxes, figure size, and note text written next to figures.
- Edit `analysis_hooks.py` to change which model-specific figures are rendered, which variables are included, and which simulation objects feed each displayed figure.
- Edit `DEQN/analysis.py` to change the top-level display order, which generated tables and figures are included in the master output, and how the combined `figures_tables_<analysis_name>.tex` wrapper is assembled.

Operational rule:

- If the issue is visual formatting, start in `tables.py` or `plot_helpers.py`.
- If the wrong object is being shown, start in `analysis_hooks.py`.
- If the wrapper includes the wrong sections or ordering, start in `DEQN/analysis.py`.

### Generated LaTeX lookup

Use the generated `.tex` files as the debugging anchor. Start from the exact file or caption you see in
`<model_dir>/analysis/<analysis_name>/figures_tables_<analysis_name>.tex`, then trace back to the writer below.

The wrapper file currently does two things:

- `DEQN/analysis.py` → `_build_analysis_latex_sections(...)` decides which generated table fragments and figure groups are included, and in what order.
- `DEQN/analysis.py` → `_write_analysis_results_latex(...)` writes the master wrapper `figures_tables_<analysis_name>.tex` and controls wrapper-only LaTeX such as page breaks, figure sizing, subfigure widths, and whether section headers are emitted.

For the current `basefinal` output, use this map:

| What you see in generated LaTeX                                                                                                      | Generated fragment file                                                 | Change the source here                                                                                     | Why this is the right place                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `\input{calibration_table_basefinal.tex}`                                                                                            | `analysis/basefinal/calibration_table_basefinal.tex`                    | `DEQN/analysis/tables.py` → `create_calibration_table(...)` and `_create_model_vs_data_moments_table(...)` | This function writes the LaTeX table body, caption, label, note text, panel titles, row labels, and column headers for the model-vs-data table.                                                                            |
| Caption `Model vs. data business cycle moments`                                                                                      | inside `calibration_table_basefinal.tex`                                | `DEQN/analysis/tables.py` → `_create_model_vs_data_moments_table(...)`                                     | The caption string is hardcoded there. If you say “modify the table title `Model vs. data business cycle moments`,” this is the function to edit.                                                                          |
| `\input{stochastic_ss_aggregates_basefinal.tex}`                                                                                     | `analysis/basefinal/stochastic_ss_aggregates_basefinal.tex`             | `DEQN/analysis/tables.py` → `create_stochastic_ss_aggregates_table(...)`                                   | This function writes the aggregate stochastic-steady-state LaTeX table, including caption, label, note, and row set.                                                                                                       |
| Caption `Aggregate stochastic steady state`                                                                                          | inside `stochastic_ss_aggregates_basefinal.tex`                         | `DEQN/analysis/tables.py` → `create_stochastic_ss_aggregates_table(...)`                                   | The caption is set directly in that writer.                                                                                                                                                                                |
| `\input{simulation/descriptive_stats_basefinal.tex}`                                                                                 | `analysis/basefinal/simulation/descriptive_stats_basefinal.tex`         | `DEQN/analysis/tables.py` → `create_descriptive_stats_table(...)`                                          | This function writes the descriptive-statistics table caption, label, note, and the displayed method/variable layout.                                                                                                      |
| Caption `Descriptive statistics`                                                                                                     | inside `simulation/descriptive_stats_basefinal.tex`                     | `DEQN/analysis/tables.py` → `create_descriptive_stats_table(...)`                                          | Edit this when changing the descriptive-statistics table title or note text.                                                                                                                                               |
| `\input{welfare_basefinal.tex}`                                                                                                      | `analysis/basefinal/welfare_basefinal.tex`                              | `DEQN/analysis/tables.py` → `create_welfare_table(...)`                                                    | This function writes the welfare table caption, label, note, and row layout.                                                                                                                                               |
| Caption `Welfare cost of business cycles`                                                                                            | inside `welfare_basefinal.tex`                                          | `DEQN/analysis/tables.py` → `create_welfare_table(...)`                                                    | Edit here for welfare-table title or note changes.                                                                                                                                                                         |
| Aggregate histogram grouped figure caption, subgroup titles, and ordering                                                            | emitted directly into `figures_tables_basefinal.tex`                    | `DEQN/analysis.py` → `_build_analysis_latex_sections(...)`                                                 | The wrapper decides the grouped histogram figure layout and places it immediately after descriptive statistics in the master output.                                                                                       |
| Aggregate histogram note text                                                                                                        | `analysis/basefinal/simulation/aggregate_histograms_basefinal_note.tex` | `analysis_hooks.py`                                                                                         | The histogram note is generated from the active nonlinear simulation workflow and benchmark simulation metadata inside the model-specific postprocess hook.                                                                  |
| Figure captions and notes such as `Aggregate consumption response to a TFP shock in ...`                                             | emitted directly into `figures_tables_basefinal.tex`                    | `DEQN/analysis.py` → `_build_analysis_latex_sections(...)`                                                 | The grouped figure captions and note text for aggregate IRs, sectoral IRs, stochastic-SS sectoral figures, and ergodic sectoral figures are assembled in the wrapper builder, not in `tables.py`.                          |
| Which figures appear, in what order                                                                                                  | `figures_tables_basefinal.tex`                                          | `DEQN/analysis.py` → `_build_analysis_latex_sections(...)`                                                 | This is where the wrapper decides the sequence: model-vs-data table, aggregate IRs, stochastic-SS sectoral figures, aggregate stochastic-SS table, descriptive stats, aggregate histograms, welfare, sectoral IRs, and ergodic sectoral figures. |
| Wrapper-only LaTeX such as figure height limits, subfigure widths, first-figure sizing, `\clearpage`, and section-header suppression | `figures_tables_basefinal.tex`                                          | `DEQN/analysis.py` → `_write_analysis_results_latex(...)`                                                  | These choices are not stored in the fragment files; they are emitted only when the master wrapper is assembled.                                                                                                            |
| The PNG image itself looks wrong: panel count, legend, axes, line styles, note written onto the image                                | `analysis/basefinal/IRs/*.png` or `analysis/basefinal/simulation/*.png` | `plot_helpers.py` and the relevant render call in `analysis_hooks.py`                                      | Change `plot_helpers.py` for the visual design of the PNG. Change `analysis_hooks.py` if the wrong variables or wrong simulation object are feeding that plot.                                                             |

Practical examples:

- If the request is “modify the table title `Model vs. data business cycle moments`,” edit `DEQN/analysis/tables.py` in `_create_model_vs_data_moments_table(...)`.
- If the request is “make the first figure larger,” edit `DEQN/analysis.py` in `_write_analysis_results_latex(...)`, because that is wrapper-only sizing logic.
- If the request is “rename the caption `Aggregate consumption response to a TFP shock in Mining, Oil and Gas.`,” edit `DEQN/analysis.py` in `_build_analysis_latex_sections(...)`.
- If the request is “change the line colors or legend on the aggregate IR PNG,” edit `plot_helpers.py`, not the LaTeX writer.

Rule of thumb:

- If the text already exists inside a generated fragment like `calibration_table_<analysis_name>.tex`, `stochastic_ss_aggregates_<analysis_name>.tex`, `simulation/descriptive_stats_<analysis_name>.tex`, or `welfare_<analysis_name>.tex`, the source is usually `DEQN/analysis/tables.py`.
- If the text exists only in `figures_tables_<analysis_name>.tex`, the source is usually `DEQN/analysis.py`.
- If the issue is with the pixels inside a PNG rather than the LaTeX around it, the source is usually `plot_helpers.py` or `analysis_hooks.py`.

## Analysis outputs

The Python layer writes outputs under:

- `<model_dir>/analysis/<analysis_name>/`
- `<model_dir>/analysis/<analysis_name>/simulation/`
- `<model_dir>/analysis/<analysis_name>/IRs/`

Main displayed exercises:

1. model vs data moments
2. aggregate IRs
3. sectoral variables in stochastic SS
4. aggregate stochastic SS
5. descriptive statistics
6. aggregate histograms
7. welfare costs
8. sectoral IRs
9. ergodic mean sectoral variables

For the exact current execution inventory, see:

- `DEQN/docs/analysis_inventory.md`

## Method appearance in tables

A useful operational distinction is:

- Some tables are driven by `analysis_variables_data`.
- Some tables are driven directly by moment dictionaries.
- Some figures are driven by `gir_data`.
- Some figures are driven by stochastic-SS objects.

For simulation-based benchmark methods to appear in descriptive statistics, they must exist as simulation blocks inside `ModelData_simulation.mat`.

Example:

- if `ModelData_simulation.mat` contains only `FirstOrder`, then only the log-linear benchmark can appear in simulation-based descriptive tables
- `PerfectForesight` and `MITShocks` cannot appear there unless those blocks are actually present in the MATLAB simulation object

This is separate from `ModelData.mat`, which may still contain benchmark summary stats in `Statistics.<Method>.ModelStats`.

## Current design rules

The current Python analysis logic follows these conventions:

- Exactly one nonlinear DEQN simulation source is active per run.
- `long_simulation = true` selects the long ergodic run; `long_simulation = false` selects the common-shock run.
- Sectoral stochastic-SS figures should use the selected nonlinear run.
- GIR exercises should use the selected nonlinear run.
- MATLAB simulation benchmarks should only appear when the corresponding method blocks exist in `ModelData_simulation.mat`.
- Theoretical moments are no longer part of the descriptive-statistics table.

## Cleaning

The analysis codebase can still be cleaned in several places.

### Redundant outputs

- `create_stochastic_ss_table` and `create_stochastic_ss_aggregates_table` currently overlap heavily for the reported aggregate block (`C`, `I`, `GDP`, `K`, `L`, and `Intratemporal Utility`).
- It would be cleaner to decide whether both are really needed.

### Dormant code paths

- `analysis_hooks.postprocess_analysis(...)` is effectively bypassed because `analysis.py` uses `prepare_postprocess_analysis(...)` plus explicit render calls.
- If the split flow is final, the old wrapper path could be removed or simplified.

### Unused plotting helpers

- `plot_combined_impulse_responses`
- `plot_gir_heatmap`
- several alternative IR panel helpers

These are still in the codebase but are not part of the current displayed pipeline.

### Config / filtering logic

- Some filtering logic is still present even when the corresponding config key is no longer part of the active config block.
- The analysis layer would be clearer if deprecated config branches were either restored intentionally or removed.

### Inventory vs implementation drift

- The code is now more structured than before, but the analysis pipeline still mixes orchestration, compatibility handling, and output rendering in `DEQN/analysis.py`.
- Moving more selection logic into hooks or dedicated helpers would make the top-level file easier to audit.

### MATLAB/Python contract checks

- The Python side assumes specific method names and field layouts in `ModelData_simulation.mat`.
- A small validation layer that prints exactly which method blocks were found would make debugging missing descriptive-stat columns much faster.

## Suggested reading order

If you want to understand the Python analysis side quickly, read in this order:

1. `DEQN/analysis.py`
2. `analysis_hooks.py`
3. `aggregation.py`
4. `DEQN/analysis/tables.py`
5. `plot_helpers.py`
6. `MATLAB/CODEBASE_README.md`
7. `MATLAB/ModelData_README.md`

That order goes from orchestration, to economics and aggregation, to rendering, and then back to the MATLAB source objects.
