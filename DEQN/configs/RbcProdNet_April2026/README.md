# RbcProdNet April 2026 Configs

This directory contains JSON overrides for reproducible `RbcProdNet_April2026`
runs. The Python entry points are:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
python -m DEQN.analysis_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

## Config Layers

The loader supports an optional `extends` field. Paths are resolved relative to
the JSON file that declares them and are merged from left to right:

```json
{
  "extends": ["../base_train.json", "../base_analysis.json"],
  "train": {
    "exper_name": "highsigmay"
  }
}
```

Use this layout for new runs:

```text
RbcProdNet_April2026/
  base_train.json
  base_analysis.json
  experiments/
    <experiment>.json
    <experiment>.resume.json
    <experiment>.ir_recover.json
    <experiment>.smoke.json
```

The older `*_runpod.json` files are still supported. They are full combined
configs kept for compatibility with existing runs and scripts.

## Merge Rules

- `train` is merged into the defaults in `DEQN/train.py`.
- `analysis` is merged into the defaults in `DEQN/analysis.py`.
- Nested dictionaries are recursively merged.
- Experiment overlays should only set values that differ from the shared base.
- The resolved config is persisted in experiment and analysis output folders.

## Naming Conventions

- `exper_name` names the training output folder under `experiments/`.
- `analysis_name` usually matches `exper_name`.
- `config_ir_finetune.exper_name` should usually be `<exper_name>_IR`.
- `ir_experiment_to_analyze` should match the IR fine-tuning experiment when IR
  figures should use the auxiliary IR checkpoint.
- Smoke configs should reduce epochs, steps, episodes, periods, Monte Carlo
  draws, and analysis simulation sizes.

## Required External Files

Training requires the configured `ModelData*.mat` file in the model folder.
Analysis can additionally use:

- `ModelData_simulation*.mat` for benchmark simulation comparisons.
- `ModelData_IRs*.mat` for benchmark impulse-response overlays.

When benchmark files are unavailable, prefer long DEQN-only simulations and make
the missing benchmark inputs explicit in the config.
