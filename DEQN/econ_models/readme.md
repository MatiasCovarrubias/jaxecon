# DEQN Economic Models

This directory contains economic model implementations used by DEQN. Each active
model folder should expose a `Model` class from `model.py`.

## Reference Models

| Model folder | Role | External data |
| --- | --- | --- |
| `RBC/` | Simple pure-Python real business cycle example | None |
| `NK/` | Simple pure-Python three-equation New Keynesian example | None |
| `RbcProdNet_April2026/` | Current production-network research model | Requires MATLAB/Dynare `.mat` artifacts |
| older `RbcProdNet_*` folders | Research history / transitional support | Varies |

Use `RBC/` or `NK/` when learning the interface. Use
`RbcProdNet_April2026/` when working on the production-network research pipeline.

## Required Model Contract

A DEQN model must provide a `Model` class with normalized state and policy
objects. The shared algorithm code calls the methods below inside JAX
transformations, so implementations should use JAX arrays and JAX-compatible
control flow.

Required attributes:

- `state_ss`: steady-state state vector in model units.
- `state_sd`: scale vector used to normalize states.
- `policies_ss`: steady-state policy/control vector in model units.
- `policies_sd`: scale vector used to normalize policies.
- `dim_states`: number of state variables.
- `dim_policies`: number of policy/control variables.

Required methods:

```python
class Model:
    def initial_state(self, rng, init_range=0):
        ...

    def step(self, state, policy, shock):
        ...

    def expect_realization(self, state_next, policy_next):
        ...

    def loss(self, state, expect, policy):
        ...

    def sample_shock(self, rng, n_draws=1):
        ...

    def mc_shocks(self, rng=None, mc_draws=8):
        ...
```

The shared DEQN loss expects:

- `initial_state` to return a normalized state vector.
- `step` to take normalized `state`, normalized `policy`, and one shock draw,
  then return the next normalized state.
- `expect_realization` to compute the next-period objects that enter the model's
  expectations.
- `loss` to return `(mean_loss, mean_accuracy, min_accuracy, mean_accs_foc,
  min_accs_foc)`.
- `sample_shock` to draw simulation shocks.
- `mc_shocks` to draw the Monte Carlo shock array used in expectations.

## Optional Analysis Contract

Models can expose analysis variables directly or through hooks:

- `get_aggregates(simul_policies, simul_states)` for simple aggregate outputs.
- `get_analysis_variables(state_logdev, policies_logdev)` when labeled variables
  can be computed directly from states and policies.
- `analysis_hooks.py` when analysis requires model-specific context,
  aggregation, benchmark adapters, or GIR state selection.
- `plots.py` with `MODEL_SPECIFIC_PLOTS` for auto-discovered model-specific
  figures.

The generic analysis layer lives in `DEQN/analysis/`. Model-specific logic should
stay in the model folder unless it is truly reusable across models.

## Adding A New Model

1. Create `DEQN/econ_models/<ModelName>/model.py`.
2. Implement the required `Model` attributes and methods.
3. Add a small training script or JSON-compatible entry point.
4. Start with the plain MLP in `DEQN/neural_nets/neural_nets.py`.
5. Add analysis hooks only after the model trains and basic simulations are
   finite.
6. Add a small smoke configuration before adding large research runs.

For a pure-Python model, follow `RBC/model.py` or `NK/model.py`. For a model that
loads externally computed steady states, normalization statistics, or linear
solutions, follow `RbcProdNet_April2026/model.py` and document the required input
objects in the model folder.