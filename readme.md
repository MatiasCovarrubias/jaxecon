# JaxEcon

JaxEcon is a JAX library of solvers for dynamic economic models. The shared
comparison object is the **one-sector RBC family** in `DEQN/econ_models/RBC/`.
Three solvers train or compute a policy on that economy: **DEQN** (Euler residual),
**APG** (pathwise policy gradient), and **TimeIteration** (grid Newton). They share
the economics, not a single Python class. Each solver has its own model
contract.

`VFI/` is an educational value-function example. `PI/` is a placeholder.

## One-sector RBC family

The default economy is a saving-rate planner problem: capital and log TFP as
states, a logit saving rate as the action, CES/CRRA utility. Shared primitives
and trainer defaults live in
[`DEQN/econ_models/RBC/train_shared.py`](DEQN/econ_models/RBC/train_shared.py)
(`SHARED_RBC_TRAIN`). DEQN and APG public trainers read that file so a
comparison run uses the same model, `PolicyNet` width, shock budget, and Adam
schedule. Time iteration uses the same parameter names in
`TimeIteration/models/rbc.py`.

Switches on the smooth `Model` (do not invent a new class for these):

| Switch | Default | Effect |
| --- | --- | --- |
| `cbar_frac` | `0.0` | Stone-Geary floor \(\bar c =\) `cbar_frac` \(\cdot C_{ss}\) |
| `phi` | `2.0` | capital-adjustment cost; `0` turns it off |
| `i_min_frac` | `0.0` | investment floor as a fraction of \(I_{ss}\) |
| `shock_sd`, `rho` | `0.02`, `0.7` | log-TFP innovation and persistence |
| `eps_c` | `0.5` | IES (risk aversion is \(1/\varepsilon_c\)) |

Constraint variants are subclasses, not flags:

| Class | File | Role |
| --- | --- | --- |
| `Model` | `model.py` | smooth interior RBC |
| `ProjectedIrreversibleModel` | `projected_irreversible_model.py` | clip \(I\) to the floor inside the step |
| `IrreversibleModel` | `irreversible_model.py` | KKT residual with a multiplier |
| `ExactKinkModel` | `exact_kink_model.py` | explicit kink in the policy map |

APG does not reimplement the economics. `APG.environments.RbcMultiSector`
wraps one of those DEQN classes and exposes the APG environment contract.
Time iteration implements the same switches on its own Layer-0 spec
(`TimeIteration/models/rbc.py`); it does not import the DEQN `Model`.

A self-contained APG economy that is not this RBC family is
`APG.environments.TimeToBuildRbc` (Kydland–Prescott time to build).

## Algorithms

| Algorithm | Role | Entry point |
| --- | --- | --- |
| [**DEQN**](DEQN/) | Neural policy trained on Euler / KKT residuals | `python -m DEQN.econ_models.RBC.train` |
| [**APG**](APG/) | Neural policy trained by differentiating the discounted return | `python -m APG.train` (smoke: `python -m APG.smoke`) |
| [**TimeIteration**](TimeIteration/) | Grid time iteration with a semi-smooth Newton local solver | `python -m TimeIteration.train` |
| [**VFI**](VFI/) | Educational value-function iteration | `python VFI/vfi.py` |
| **PI** | Planned policy-iteration placeholder | — |

DEQN also has a production-network research pipeline
(`DEQN/train.py`, `DEQN/econ_models/RbcProdNet_April2026/`) that needs MATLAB /
Dynare `.mat` files. That is a separate model family, not the comparison RBC.

## Solver contracts

To put an economy on a solver, implement that solver's contract. To put the
same economy on all three, write it three times (or wrap, as `RbcMultiSector`
does for APG). Full checklists live in the package READMEs; the surfaces are:

**DEQN** — a `Model` in `DEQN/econ_models/<Name>/model.py`. Required:
`state_ss`, `state_sd`, `policies_ss`, `policies_sd`, `dim_states`,
`dim_policies`, plus `initial_state`, `step`, `expect_realization`, `loss`,
`sample_shock`, `mc_shocks`. Details:
[DEQN/econ_models/readme.md](DEQN/econ_models/readme.md). Follow
`DEQN/econ_models/RBC/model.py`.

**APG** — a `WelfareEnvironment`: `initial_state`, `sample_shock`,
`transition`, `training_reward`, `terminal_value`, `discount_rate`,
`value_ss`, `obs_ss`, `action_dim`, and the welfare helpers. Coordinates are
normalized deviations from the deterministic steady state. Verify with
`APG.environments.check_environment(env)` and, in float64,
`APG.loglinear.solve.linearize_environment(env)` (FOC residual at the steady
state must be zero). Details: [APG/README.md](APG/README.md). Follow
`TimeToBuildRbc` for a from-scratch environment, or wrap a DEQN `Model` as
`RbcMultiSector` does.

**TimeIteration** — Layer 0 only (`TimeIteration/models/protocol.py`):
`steady_state`, `exog_process`, `endo_grid`, `transition`, `auxiliary`,
`expectand`, `arbitrage`, `initial_policy`. Adding an unknown appends a row
to `arbitrage` and a component of `x`. Details:
[TimeIteration/README.md](TimeIteration/README.md). Follow
`TimeIteration/models/rbc.py`.

## Quick Start

The public examples below do not require private MATLAB or Dynare artifacts.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

```bash
python -m DEQN.econ_models.RBC.train    # DEQN on the shared RBC
python -m APG.smoke                     # APG component check (~5 s)
python -m APG.train                     # APG on the shared RBC (minutes on CPU)
python -m TimeIteration.train           # closed-form gate, then a small irreversible RBC
python VFI/vfi.py                       # educational VFI
```

`APG/train.py` uses the shared research settings (512-period rollouts). Shrink
`n_epochs` / `periods_per_epis` for a local smoke, or stay with `APG.smoke`.
See [DEVELOPMENT.md](DEVELOPMENT.md) for tests and GPU notes.

```bash
python -m unittest discover -s APG/tests -t .
python -m unittest discover -s DEQN/tests -t .
python -m unittest TimeIteration.tests.test_closed_form TimeIteration.tests.test_engine
```

## Other models

| Model | Solver | Data |
| --- | --- | --- |
| `DEQN/econ_models/NK/` | DEQN | none (three-equation NK) |
| `APG/environments/time_to_build.py` | APG | none |
| `DEQN/econ_models/RbcProdNet_April2026/` | DEQN | MATLAB/Dynare `ModelData*.mat` |

The production-network pipeline is not a no-data quick start. See
[DEQN/readme.md](DEQN/readme.md),
[TRAINING_README.md](DEQN/econ_models/RbcProdNet_April2026/TRAINING_README.md),
and
[ANALYSIS_README.md](DEQN/econ_models/RbcProdNet_April2026/ANALYSIS_README.md).

## Repository Structure

```text
jaxecon/
├── DEQN/
│   ├── algorithm/       # DEQN simulation, residual loss, training, evaluation
│   ├── analysis/        # Generic analysis utilities (RbcProdNet-oriented)
│   ├── configs/         # JSON overlays for RbcProdNet runs
│   ├── econ_models/     # DEQN Model implementations (RBC family, NK, RbcProdNet)
│   ├── neural_nets/     # PolicyNet and loglinear-baseline nets (shared with APG)
│   ├── training/        # Checkpoints and experiment helpers
│   ├── train.py         # RbcProdNet training entry point
│   └── analysis.py      # RbcProdNet analysis entry point
├── APG/
│   ├── algorithm/       # Pathwise rollout, return loss, eval, welfare
│   ├── environments/    # WelfareEnvironment, RbcMultiSector, TimeToBuildRbc
│   └── train.py         # APG on the shared RBC
├── TimeIteration/       # Grid time iteration; Layer 0 in models/
├── VFI/                 # Educational value-function iteration
└── PI/                  # Planned policy-iteration work
```

## Configuration

Most public trainers keep an editable `config` dictionary near the top. The
shared RBC defaults are `SHARED_RBC_TRAIN` in
`DEQN/econ_models/RBC/train_shared.py`; override keys there or in the trainer
script. Reproducible RbcProdNet runs can also use JSON configs:

```bash
python -m DEQN.train_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
python -m DEQN.analysis_importconfig --config DEQN/configs/RbcProdNet_April2026/highsigmam_smoke_runpod.json
```

The JSON path is for unattended runs. Script-level configs remain the
canonical defaults for interactive work.

## Requirements

- Python 3.10-3.13
- JAX 0.7+
- Flax 0.8+
- Optax 0.2+
- Orbax checkpointing for saved DEQN/APG experiments

## License

MIT
