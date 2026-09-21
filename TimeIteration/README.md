# TimeIteration

A model-agnostic global solution method: time iteration with a semi-smooth Newton
local solver. The model enters through a residual. The solver loop does not.

## Quick start

```bash
source .venv/bin/activate
python -m TimeIteration.train
python -m unittest TimeIteration.tests.test_closed_form TimeIteration.tests.test_engine
```

The smoke run first checks the closed form `s = ??` under log utility and `? = 1`,
then solves a small irreversible RBC.

## Layers

| Layer | Role |
| --- | --- |
| 0 | Model spec: `Params`, grids, transition, auxiliary, `arbitrage` |
| 1 | 1-D interpolation in capital; exogenous states stay discrete |
| 2 | Expectation as one `einsum` against the Rouwenhorst row |
| 3 | Damped Newton with `jax.jacfwd` on the per-point residual |
| 4 | Time iteration to a policy array of shape `(n_a, n_k, n_x)` |
| 5 | Simulation, Euler errors (same `arbitrage`), welfare, bind set |
| 6 | Implicit differentiation of the fixed point w.r.t. `Params` |

A new model writes Layer 0 only. Adding an unknown (labor, a sector) appends a
row to `arbitrage` and a component of `x`.

## Closed-form gate

Log utility and `? = 1` give a constant saving rate `??` with the constraint
off. The pipeline must recover that to machine precision before irreversibility
is turned on. A slack floor (`I_min` low enough never to bind) must return the
same policy and a zero multiplier.

## Configuration

`Params` is a JAX pytree of float64 scalars. Grid sizes live in `GridSpec` and
are static. Enable float64 before any JAX work; the package does this on import.

Warm start from the deterministic steady-state control. Optional Anderson
acceleration is available through `solve(..., anderson_memory=5)` if the outer
loop stalls.
