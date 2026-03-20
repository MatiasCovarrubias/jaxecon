# DEQN analysis pipeline — RbcProdNet_March2026 (inventory)

Scope: what `DEQN/analysis.py` **actually runs** for a model with hooks like `RbcProdNet_March2026`. Output roots:  
`<model_dir>/analysis/<analysis_name>/` (tables, `config.json`), `.../simulation/` (simulation-related TeX + sector plots), `.../IRs/` (IR PNGs).

---

## 1. Display order (as executed in `main()`)

| # | Exercise | Brief description | Where computed | Table / figure output |
|---|----------|-------------------|----------------|----------------------|
| 0 | Setup + roll-forward | Load `ModelData`, build `Model`, optional Dynare sim IRF/schedule, train checkpoints, episode + optional common-shock sim, welfare CE sample, stochastic SS, GIR object | `analysis.py`; `simul_analysis.py`; `stochastic_ss.py`; `GIR.py`; `welfare.py`; `model_hooks.py`; hooks `compute_analysis_variables` / `prepare_analysis_context` | `config.json` in `analysis_dir` |
| 1 | Model vs data moments | Empirical targets vs model moments (MATLAB `ModelStats` + nonlinear moments from sim + optional Dynare sim aggregates) | `aggregation.py` + hooks `prepare_postprocess_analysis` → `_build_calibration_method_stats`; first-order stats from `md["Statistics"]` | LaTeX: `analysis_dir/calibration_table_<analysis_name>.tex` (console from `tables.create_calibration_table`) |
| 2 | Aggregate IRs | Sector TFP shocks; DEQN IR (`GIR`/`IR_stoch_ss`) vs MATLAB benchmark (`ModelData_IRs`) | `GIR.py` (`create_GIR_fn`); MATLAB via `matlab_irs` + hooks `_build_ir_render_context` | PNG: `IRs/IR_<var>_<sector>_<analysis_name>.png` via `plot_helpers.plot_sector_ir_by_shock_size` |
| 3 | Sectoral variables in stochastic SS | Bar charts of K,L,Y,M,Q (% dev from det. SS) at stochastic SS, optional upstreamness ρ | `stochastic_ss.py` + hooks `render_sectoral_stochss_outputs` | PNG: `simulation/sectoral_<k|l|y|m|q>_stochss_<analysis_name>.png` |
| 4 | Aggregate stochastic SS | Same four aggregates’ **levels** at stochastic SS (C,I,GDP,K) | `stochastic_ss.py` → `compute_analysis_variables` at SS | LaTeX: `stochastic_ss_aggregates_<analysis_name>.tex` **and** `stochastic_ss_<analysis_name>.tex` (both use the same four series; aggregates table respects `stochss_methods_to_include`) |
| 5 | Descriptive statistics (simulation) | Mean/sd/skew/excess kurt for **configured** variables on ergodic (and benchmark) series | Time series in `analysis_variables_data` after hooks (`process_simulation_with_consistent_aggregation`, NN sim) | LaTeX: `simulation/descriptive_stats_<analysis_name>.tex` |
| 5b | Ergodic aggregate moments | Same four aggregates, stats table | From filtered `analysis_variables_data` | LaTeX: `simulation/ergodic_aggregate_stats_<analysis_name>.tex` |
| 5c | Comparative descriptive (optional) | All variables side-by-side if **≥2** methods after filters | Same series as 5 | LaTeX: `simulation/descriptive_stats_comparative_<analysis_name>.tex` (only if `len(filtered...) > 1`) |
| 6 | Welfare cost of BC | CE % from simulated utilities (NN + optional Dynare paths) | `welfare.py` + `_compute_welfare_cost_from_sample` / `_welfare_cost_from_dynare_simul` | LaTeX: `analysis_dir/welfare_<analysis_name>.tex` |
| 7 | Sectoral IRs | Same IR machinery as (2), sectoral variable list from config | `GIR.py` + hooks `render_sectoral_ir_outputs` | PNG: `IRs/IR_<var>_<sector>_<analysis_name>.png` |
| 8 | Ergodic mean sectoral levels | Time-mean sector K,L,Y,M,Q (% dev from det. SS), ergodic runs only | Raw sim in `raw_simulation_data` | PNG: `simulation/sectoral_<k|l|y|m|q>_ergodic_<analysis_name>.png` |

**Note:** In code, **aggregate stochastic SS tables (4)** come **before** descriptive statistics **(5)**; **ergodic sectoral bars (8)** run **after** sectoral IRs **(7)**, not before welfare.

---

## 2. Script-by-script

### `DEQN/analysis.py`

- Orchestrates config, I/O paths, experiment loop, postprocess hook, ordered sections above.
- Saves `config.json`; writes all `create_*_table` calls; calls hook `render_*` functions.

### `DEQN/analysis/tables.py`

- `create_calibration_table` → model vs data TeX + console.
- `create_stochastic_ss_aggregates_table`, `create_stochastic_ss_table` → aggregate SS TeX.
- `create_descriptive_stats_table` → variable-wise descriptive TeX + console.
- `create_ergodic_aggregate_stats_table` → aggregate-only descriptive TeX.
- `create_comparative_stats_table` → wide comparative TeX (conditional).
- `create_welfare_table` → welfare TeX.

### `DEQN/analysis/simul_analysis.py`

- `create_episode_simulation_fn_verbose`, `simulation_analysis` — long ergodic NN paths + `analysis_variables`.
- `create_shock_path_simulation_fn`, `simulation_analysis_with_shocks` — common-shock path.
- `compute_analysis_dataset_with_context` — recompute analysis vars under fixed `analysis_context`.

### `DEQN/analysis/stochastic_ss.py`

- `create_stochss_fn`, `create_stochss_loss_fn` — find stochastic SS state/policy from ergodic sample; accuracy loss.

### `DEQN/analysis/GIR.py`

- `create_GIR_fn` — IR paths for configured shock sizes / states (`config["ir_methods"]`).

### `DEQN/analysis/welfare.py`

- `get_welfare_fn` — expectation of discounted utility over simulated trajectories.

### `DEQN/analysis/model_hooks.py`

- `load_model_analysis_hooks`, `apply_model_config_defaults`, `compute_analysis_variables`, `prepare_analysis_context`, `run_model_postprocess` (fallback if no `prepare_postprocess_analysis` on hook module).

### `DEQN/analysis/matlab_irs.py`

- Loaded indirectly: MATLAB IR structures for benchmark curves in plots (via `plot_helpers` / `RbcProdNet_March2026/matlab_irs`).

### `DEQN/econ_models/RbcProdNet_March2026/analysis_hooks.py`

- `prepare_postprocess_analysis` — ergodic prices, SS corrections, Dynare simulations merged into `analysis_variables_data`, `calibration_method_stats`, `postprocess_context`.
- `render_aggregate_ir_outputs`, `render_sectoral_stochss_outputs`, `render_sectoral_ir_outputs`, `render_ergodic_sectoral_outputs` — figures only (no TeX).
- `postprocess_analysis` — **not** called by `analysis.py` when `prepare_postprocess_analysis` exists (would duplicate renders if used instead of the split path).

### `DEQN/econ_models/RbcProdNet_March2026/plot_helpers.py`

- `plot_sector_ir_by_shock_size` — IR PNGs.
- `plot_sectoral_variable_stochss`, `plot_sectoral_variable_ergodic` — sectoral bar PNGs.
- Other helpers (`plot_combined_impulse_responses`, `plot_ergodic_histograms`, `plot_gir_heatmap`, `plot_impulse_responses` grid, `IRPanel_*`, `CombinedIR_*`, capital-specific plots) — **not** invoked by current `analysis.py` flow.

### `DEQN/econ_models/RbcProdNet_March2026/plots.py`

- `MODEL_SPECIFIC_PLOTS` is **empty**; optional extra PNGs only if entries are added.

### `DEQN/econ_models/RbcProdNet_March2026/aggregation.py` (and related)

- Simulation processing, ergodic prices, moments for calibration columns (`process_simulation_with_consistent_aggregation`, `compute_model_moments_with_consistent_aggregation`, etc.) — used from hooks.

### `DEQN/analysis/eval.py`

- **Not** referenced by `analysis.py` in this pipeline.

---

## 3. Not used in this pipeline (or redundant)

**Analyses / code paths**

- Hook `postprocess_analysis` entrypoint (for this model): `analysis.py` uses `prepare_postprocess_analysis` + separate `render_*`; `postprocess_analysis` only runs if something calls it manually.
- `theoretical_stats` in hooks: still returned but currently empty for March2026 after theoretical block removal — nothing consumes it in descriptive tables.
- `DEQN/analysis/eval.py` — unused by main script.
- Duplicate SS reporting: `stochastic_ss_aggregates_*` and `stochastic_ss_*` tables both cover C,I,GDP,K at stochastic SS (second filterable by `stochss_methods_to_include`).

**Tables / figures defined but not produced by `analysis.py` for March2026**

- Histograms: `plot_ergodic_histograms` (no call).
- `plot_combined_impulse_responses` → `CombinedIR_*.png` (no call).
- Multi-panel `IRPanel_*.png` from panel builder in `plot_helpers` (no call from hook render path).
- `plot_gir_heatmap` (no call).
- Generic `plot_impulse_responses` in `DEQN/analysis/plots.py` (not used; NK model has its own).
- `MODEL_SPECIFIC_PLOTS` outputs — none while registry empty.
- Filename patterns in `plot_helpers` for `GIR_*_*.png` from the grid helper — not used by `plot_sector_ir_by_shock_size` (that uses `IR_*.png`).

---

*Generated from repository layout as of the inventory pass; filenames follow `tables.py` rewrites when `analysis_name` is set.*
