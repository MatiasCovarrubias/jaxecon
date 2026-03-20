# MATLAB Codebase Guide

This folder contains the runtime pipeline for the March 2026 RBC production-network model.

For saved-object documentation, see `ModelData_README.md`.

## Runtime Map

Read these files in order:

1. `main.m`
2. `runtime_config.m`
3. `params_config.m`
4. `calibration/load_calibration_data.m`
5. `steady_state/calibrate_steady_state.m`
6. `dynare/run_dynare_analysis.m`
7. `utils/run_irf_loop.m`
8. `utils/build_ModelData.m`
9. `utils/build_ModelData_simulation.m`
10. `utils/build_ModelData_IRs.m`
11. `utils/validate_pipeline_outputs.m`
12. `utils/save_experiment_results.m`

## `main.m`

`main.m` is the canonical entry point. It does the following:

1. Sets paths with `setup_runtime_paths`.
2. Checks required input files.
3. Loads `config` from `runtime_config.m`.
4. Loads `params` from `params_config.m`.
5. Creates the experiment folder from `config.date + config.exp_label`.
6. Loads calibration inputs through `load_calibration_data`.
7. Loads or builds the steady state.
8. Runs Dynare simulations.
9. Runs the IRF loop if requested.
10. Packages outputs into `ModelData`, `ModelData_simulation`, and `ModelData_IRs`.
11. Validates and optionally saves results.

## Configuration

The user-facing config files are:

- `runtime_config.m`
- `params_config.m`

Each file has a `use_defaults` boolean:

- `true`: load `runtime_config_defaults.m` or `params_config_defaults.m`
- `false`: use the values written in the user-facing file

Use:

- `runtime_config.m` for experiment settings, Dynare flags, horizons, burn-in, burn-out, saving, and shock sizes
- `params_config.m` for model parameters

The compatibility wrappers:

- `utils/build_default_config.m`
- `utils/build_default_params.m`

mirror the defaults files.

### Fast test run

For a short boundary run, set in `runtime_config.m`:

- `use_defaults = false`
- `save_results = false`
- `force_recalibrate = false`
- `run_firstorder_simul = true`
- `run_secondorder_simul = false`
- `run_firstorder_irs = false`
- `run_secondorder_irs = false`
- `run_pf_irs = false`
- `run_pf_simul = false`
- `run_mit_shocks_simul = false`
- `simul_T = 10`
- `simul_burn_in = 0`
- `simul_burn_out = 0`

For the fastest rerun, reuse the cached experiment from this session:

- `config.date = "_Mar_2026"`
- `config.exp_label = "_matlab_smoke"`
- experiment folder: `experiments/Mar_2026_matlab_smoke/`
- keep `force_recalibrate = false`

There is also a cached smoke-suite steady state in:

- `experiments/Mar_2026_smoke_tests/`

Run from MATLAB or:

```bash
"/Applications/MATLAB_R2025b.app/bin/matlab" -batch "main"
```

Dynare should resolve to:

- `/Applications/Dynare/6.5-arm64/matlab/dynare.m`

## Folder Map

- `calibration/`: loads prepared inputs and builds runtime calibration objects
- `steady_state/`: steady-state calibration and solver code
- `dynare/`: Dynare runtime setup, execution, and raw result processing
- `utils/`: configuration helpers, packaging, validation, and save logic
- `plotting/`: figure generation and post-run analysis
- `testing/`: local test helpers and fixture tools
- `experiments/`: saved runs, cached steady states, logs, and fixture bundles

## Main Functions by Role

### Configuration and orchestration

- `main.m`
- `runtime_config.m`
- `params_config.m`
- `utils/build_dynare_opts.m`
- `utils/setup_experiment_folder.m`

### Calibration and steady state

- `calibration/load_calibration_data.m`
- `steady_state/calibrate_steady_state.m`

### Dynare execution

- `dynare/run_dynare_analysis.m`
- `dynare/setup_dynare_runtime.m`

### IRF processing

- `utils/run_irf_loop.m`
- `dynare/process_sector_irs.m`
- `utils/process_ir_data.m`

### Packaging and validation

- `utils/build_ModelData.m`
- `utils/build_ModelData_simulation.m`
- `utils/build_ModelData_IRs.m`
- `utils/validate_pipeline_outputs.m`
- `utils/save_experiment_results.m`

## Testing Entry Points

- `run_smoke_tests.m`: broad regression suite
- `run_local_smoke_tests.m`: fast local check
- `export_test_fixtures.m`: build reusable fixture bundle
- `run_fixture_tests.m`: replay packaging and validators from fixtures
- `run_debug_loop.m`: logged wrapper around the local test workflow

To include the thin Dynare smoke in the local test:

```matlab
cfg = build_test_defaults();
cfg.run_dynare_smoke = true;
run_local_smoke_tests(cfg)
```

## Core Conventions

- Experiment folders are named `config.date + config.exp_label`.
- The three saved objects are `ModelData`, `ModelData_simulation`, and `ModelData_IRs`.
- Simulation artifacts live in `ModelData_simulation`.
- IRF artifacts live in `ModelData_IRs.shocks`.
- IRFs use the shared 27-row layout defined by `utils/process_ir_data.m`.
