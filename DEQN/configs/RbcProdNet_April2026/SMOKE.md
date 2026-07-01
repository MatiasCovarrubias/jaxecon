# RbcProdNet Smoke Runs

These smoke runs validate the production-network path. They require staged
MATLAB/Dynare artifacts and are separate from the public no-data examples.

## Required Files

Place the configured files in `DEQN/econ_models/RbcProdNet_April2026/`.

For `highsigmam_smoke_runpod.json`:

- `ModelData_highsigmam.mat`
- `ModelData_IRs_highsigmam.mat` for IR benchmark overlays
- `ModelData_simulation_highsigmam.mat` for common-shock benchmark comparison

For the layered `experiments/highsigmay.smoke.json` example:

- `ModelData_highsigmay.mat`
- `ModelData_IRs_highsigmay.mat`
- `ModelData_simulation_highsigmay.mat`

Training requires the `ModelData*.mat` file. Analysis may be reduced when
benchmark simulation or IR files are unavailable, but the missing inputs should
be made explicit in the config.

## Legacy Full Config Smoke

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

This runs a one-epoch train, a one-epoch IR fine-tune, and a reduced analysis.

## Layered Config Smoke

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/experiments/highsigmay.smoke.json
```

This uses `base_train.json`, `base_analysis.json`, and the experiment overlay.
Use this pattern for new smoke configs.

## Expected Outcome

A successful production smoke should:

- load the configured `ModelData*.mat` object;
- create a baseline experiment under `experiments/<exper_name>/`;
- create an IR experiment under `experiments/<exper_name>_IR/` when IR
  fine-tuning is enabled;
- write `results.json` and at least one checkpoint for each trained experiment;
- write a reduced analysis folder when `run_analysis = true`;
- fail early with a clear missing-file message when required `.mat` files are
  not staged.
