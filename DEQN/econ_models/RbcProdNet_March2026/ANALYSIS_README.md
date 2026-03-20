# Python Analysis Guide

This document describes the Python-side analysis layer for `RbcProdNet_March2026`.

It complements the MATLAB documentation:

- `MATLAB/CODEBASE_README.md` explains the MATLAB runtime pipeline that builds the saved objects.
- `MATLAB/ModelData_README.md` explains the structure of `ModelData.mat`, `ModelData_simulation.mat`, and `ModelData_IRs.mat`.

This file is about what Python does *after* those objects exist.

## Scope

The Python analysis layer has one main job:

1. Load MATLAB benchmark objects and trained DEQN checkpoints.
2. Run the nonlinear DEQN simulation exercises.
3. Reconstruct analysis variables under a consistent aggregation scheme.
4. Produce the final tables and figures shown in the analysis output folders.

Canonical entry point:

- `DEQN/analysis.py`

Model-specific extension points:

- `analysis_hooks.py`
- `aggregation.py`
- `plot_helpers.py`

## Relationship with MATLAB

The Python analysis code depends on MATLAB outputs in two different ways:

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
5. Build the long nonlinear simulation runner.
6. Optionally build the auxiliary common-shock nonlinear runner from the shared MATLAB shock path.
7. For each analyzed experiment:
   - run the long nonlinear simulation
   - compute stochastic SS from that simulation
   - optionally run the common-shock nonlinear comparison
   - compute GIR objects
   - compute welfare inputs
8. Call `analysis_hooks.prepare_postprocess_analysis(...)`.
9. Render tables and figures in the analysis order.

## Rolling change log

Keep only the latest three entries here. Add newest first. Keep each entry to one short bullet focused on the behavioral change, not the implementation details.

- Arithmetic mean prices in levels now define the fixed aggregation weights in the common-shock pipeline; the nonlinear benchmark, Dynare reaggregation, and MATLAB IR reaggregation all use level-averaged prices from the active window.

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

- the long ergodic simulation is the main nonlinear analysis object
- the common-shock simulation is an auxiliary comparative object

### `DEQN/analysis/stochastic_ss.py`

This computes stochastic steady states from simulated paths and evaluates equilibrium accuracy.

For this model, the stochastic-SS sectoral figures should use the stochastic SS coming from the long ergodic simulation, not the common-shock run.

### `DEQN/analysis/GIR.py`

This computes GIR or stochastic-SS impulse responses depending on `config["ir_methods"]`.

For this model, the IR exercises are anchored to the main long ergodic run.

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

This is the model-specific bridge between generic DEQN analysis utilities and `RbcProdNet_March2026`.

Main responsibilities:

- prepare model-specific aggregation context
- compute ergodic prices from the nonlinear simulation
- recenter series using ergodic steady-state corrections
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
6. welfare costs
7. sectoral IRs
8. ergodic mean sectoral variables

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

- The long nonlinear simulation is the main nonlinear object.
- The common-shock nonlinear simulation is auxiliary and mainly used for comparative exercises such as descriptive comparisons and welfare comparisons.
- Sectoral stochastic-SS figures should use the long ergodic run.
- GIR exercises should use the long ergodic run.
- MATLAB simulation benchmarks should only appear when the corresponding method blocks exist in `ModelData_simulation.mat`.
- Theoretical moments are no longer part of the descriptive-statistics table.

## Cleaning

The analysis codebase can still be cleaned in several places.

### Redundant outputs

- `create_stochastic_ss_table` and `create_stochastic_ss_aggregates_table` currently overlap heavily for C, I, GDP, and K.
- It would be cleaner to decide whether both are really needed.

### Dormant code paths

- `analysis_hooks.postprocess_analysis(...)` is effectively bypassed because `analysis.py` uses `prepare_postprocess_analysis(...)` plus explicit render calls.
- If the split flow is final, the old wrapper path could be removed or simplified.

### Unused plotting helpers

- `plot_ergodic_histograms`
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
