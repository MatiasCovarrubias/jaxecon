# Analytical Policy Gradient (APG)

APG trains a neural-network policy by differentiating the discounted return of
a rollout through the environment's step and reward functions. The environment
must be differentiable; the gradient is exact (pathwise), not a REINFORCE
estimate.

The implementation is experimental. Its purpose in this repository is a
like-for-like comparison with DEQN on the one-sector RBC model, including
variants with an irreversibility constraint on investment.

## Quick Start

All commands run from the repository root with the `.venv` from
[`DEVELOPMENT.md`](../DEVELOPMENT.md) activated.

```bash
python -m APG.smoke                                   # ~5 s component check
python -m unittest discover -s APG/tests -t .         # ~15 s, contract, rollout, loglinear, constrained
python -m APG.train                                   # baseline APG on the shared RBC
python -m APG.constrained.train --irreversible        # constrained variants
python -m APG.train_time_to_build                     # Kydland-Prescott time to build, LQ-residual network
```

`APG/train.py` uses the shared research settings (512-period rollouts, 16
episodes per step, 20 epochs) and takes several minutes on CPU. For a short
local run, shrink the config before calling `main()`:

```python
import APG.train as t
t.config.update({"n_epochs": 1, "steps_per_epoch": 1, "epis_per_step": 2,
                 "periods_per_epis": 8, "eval_n_epis": 2, "eval_periods_per_epis": 8,
                 "welfare_n_epis": 2, "welfare_horizon": 8})
t.config = t.with_derived_counts(t.config)
t.main()
```

## Structure

```
APG/
├── train.py                 # Baseline trainer: PolicyNet or ActorCritic on the shared RBC
├── train_time_to_build.py   # Contract-only trainer for TimeToBuildRbc (train(config) for experiment scripts)
├── smoke.py                 # Tiny end-to-end check of the core components
├── algorithm/               # Core APG, environment-agnostic
│   ├── simulation.py        # Differentiable rollout (create_episode_simul_fn)
│   ├── loss.py              # -return (+ GAE value loss) per episode (create_episode_loss_fn)
│   ├── epoch_train.py       # vmap over episodes, pmean gradients, Adam step (create_epoch_train_fn)
│   ├── eval.py              # Per-epoch eval and many-rollout convergence diagnostic
│   └── welfare.py           # Common-random-number welfare rollout of any policy (create_welfare_fn)
├── environments/
│   ├── base.py              # WelfareEnvironment interface and check_environment()
│   ├── RbcMultiSector.py    # Wrapper around DEQN.econ_models.RBC models
│   └── time_to_build.py     # Kydland-Prescott (1982) time-to-build economy
├── loglinear/               # In-house LQ / Riccati log-linear policy
│   ├── solve.py             # Jacobians, Hessians, Riccati, stationary SDs
│   └── baseline.py          # Pre-training hook and PolicyNet with C
├── neural_nets/
│   └── neural_nets.py       # ActorCritic and PolicyNetLoglinear
├── training/
│   ├── run_experiment.py    # Compile, train, eval, Orbax checkpoint, results.json
│   └── plots.py             # Training-curve and LR-schedule plots
├── constrained/             # Irreversible-investment variants (RBC-specific)
│   ├── train.py             # Trainer with investment_constraint_mode switch
│   ├── naive_projected.py   # Projected investment: diagnostics and gate decision
│   ├── primal_dual.py       # Learned multiplier network, augmented Lagrangian
│   ├── exact_kink.py        # Explicit kink policy map with Euler regularizer
│   ├── networks.py          # MultiplierNet, GridMultiplierNet
│   └── run_experiment.py    # Runner for the two-network primal-dual case
├── tests/                   # Contract, rollout, and constrained-variant tests
├── DGC/                     # Unrelated research note (differentiable communication games)
└── results/                 # Run outputs, gitignored
```

`PolicyNet` is imported from `DEQN.neural_nets.neural_nets`; APG and DEQN train
the same architecture.

## Algorithm

For one episode: draw an initial state and a shock path, run
`periods_per_epis` steps of `action = net(obs)`, `reward = env.training_reward`,
`obs = env.transition(obs, action, shock)` inside `lax.scan`, and accumulate the
discounted return. The actor loss is minus that return. Gradients flow through
the whole rollout. One optimizer step averages the episode gradient over
`epis_per_step` episodes; an epoch is `steps_per_epoch` such steps.

Options in `config`:

- `use_terminal_value`: add a critic head (`ActorCritic`). The return is
  bootstrapped with `critic(last_obs) * env.value_ss`, and a GAE value loss
  (`gae_lambda`) is added. Critic inputs are detached during the rollout.
- `use_model_terminal_value`: bootstrap with the model's deterministic
  fixed-saving continuation (`env.terminal_value`, horizon
  `terminal_value_horizon`). Cannot be combined with a learned critic.
- `antithetic_episodes`: run each shock path and its negation from the same
  initial state and average the two returns.
- `rematerialize_rollout`: `jax.checkpoint` each period (forced for rollouts of
  1024 periods or more).
- `loglinear_baseline`: before training, form the LQ approximation of the
  environment at the deterministic steady state, solve the discounted Riccati
  equation for a linear policy `C`, and use it as the network baseline (same
  residual architecture as `DEQN/neural_nets/with_loglinear_baseline.py`). `C`
  is computed in Python from `training_reward` and `transition`; it is not
  loaded from Dynare. `loglinear_normalize` (default True when the baseline is
  on) writes the linear-solution standard deviations into `state_sd` and
  `policies_sd`. `loglinear_solver` is `"scipy"` or `"iterate"`.

Policy-only APG needs a long rollout. With `periods_per_epis=16` and no critic,
the actor loss improves by consuming capital. With the shared 512-period
settings, capital stays at steady state, the saving rate averages `s_ss`, and a
+0.05 TFP shock raises investment by about 10%, in line with DEQN. Judge a run
by simulated `K/K_ss`, `s`, Euler accuracy, and welfare, not by actor loss alone.

## Comparison with DEQN

`APG/train.py` and `DEQN/econ_models/RBC/train.py` both read
`DEQN.econ_models.RBC.train_shared.SHARED_RBC_TRAIN`: same model parameters,
`PolicyNet` width `[16, 16]`, shock budget (`periods_per_epis=512`,
`epis_per_step=16`, `steps_per_epoch=5`, `n_epochs=20`), Adam with cosine decay
(`learning_rate=0.025`, `cosine_alpha=0.01`), and parameter initialization at
`state_ss`. The learning-rate grid is `lr_halves_grid`: five halvings ending at
`0.05`.

The only algorithmic difference is the loss. DEQN takes one Euler-residual loss
over all simulated periods of a step (shuffled, `n_batches=1`); APG takes one
return loss over the same episodes with gradients through the rollout.

Both trainers print the same diagnostics from `DEQN.econ_models.RBC`:

- `euler_eval`: Monte Carlo Euler residual loss, mean and minimum accuracy.
- `welfare_eval`: finite-horizon discounted utility from the deterministic
  steady state and its consumption equivalent. `CE vs SS` is against the
  no-shock steady state; `CE vs s_ss` is against the same shocks with the
  steady-state saving rate. Positive is a gain. This is not the RbcProdNet
  Lucas-cost pipeline.

## Environment Interface

An environment is a plain Python object; the economics can live inside it
(`TimeToBuildRbc`) or be delegated to a model class (`RbcMultiSector`). The
core algorithm (`algorithm/`, `training/run_experiment.py`) uses only:

| Member | Used for |
| --- | --- |
| `initial_state(rng, init_range)` | episode start; `init_range=0` must return the steady state |
| `sample_shock(rng)` | one period's shock, shape `(n_shocks,)`, scaled by `simul_vol_scale` |
| `transition(state, action, shock)` | differentiable step |
| `training_reward(state, action)` | differentiable period reward, a scalar |
| `terminal_value(state, horizon)` | model continuation when `use_model_terminal_value` |
| `discount_rate`, `value_ss` | return accumulation and critic scaling |
| `obs_ss`, `action_dim` | network initialization |

`environments/base.py` declares these as `WelfareEnvironment`, plus
`deterministic_steady_state_action`, `deterministic_steady_state_welfare`, and
`consumption_equivalent`, which the welfare diagnostics use. The rollout
`vmap`s over episodes, so these methods are called on one state and one
action at a time; the `...` in the shapes below is only needed for diagnostics
that pass whole simulations.

Conventions the table does not show, all of which the existing environments
follow:

- **Coordinates.** `state` is the network input: deviations from the
  deterministic steady state divided by `state_sd`, so the steady state is the
  zero vector. `action` is the network output: deviations from
  `policies_ss` in units of `policies_sd`, so `deterministic_steady_state_action()`
  is the zero vector. The environment maps actions into economic quantities
  through bounded functions (sigmoids into `(min, max)` intervals) so that
  consumption and the other flows stay strictly positive for any real-valued
  action; `check_environment` probes actions of size `±1e3`.
- **Scales.** Expose `state_sd`, `policies_sd`, and
  `set_scales(state_sd, policies_sd)`. The log-linear baseline
  (`loglinear/solve.py`) sets both to one while differentiating, solves the LQ
  problem in those unit-scale coordinates, and, with `loglinear_normalize`,
  writes the stationary standard deviations back so the network sees z-scores.
  The baseline `action = C @ state` is only meaningful because both vectors
  are zero at the steady state.
- **`obs_ss`** is a shape and dtype template for `jnp.zeros_like` at network
  initialization; its values are not read. `RbcMultiSector` stores the
  unnormalized steady-state logs there; `TimeToBuildRbc` does the same.
- **Shapes.** `state` is `(..., state_dim)`, `action` is `(..., action_dim)`,
  `shock` is `(..., n_shocks)`. `transition` returns `(..., state_dim)` and
  `training_reward` returns `(...)`; with no leading dimension the reward is a
  scalar, which the rollout requires. Index coordinates with
  `x[..., i:i + 1]` to support both cases.
- **Tail.** `terminal_value(state, horizon)` is the discounted utility of
  applying `deterministic_steady_state_action()` for `horizon` periods from
  `state` under conditional-mean (zero) shocks. It must be differentiable in
  `state`. The choice of tail is a design quantity in the paper; any other
  tail should be a separate, named option rather than a change to this one.
- **Welfare helpers.** `deterministic_steady_state_welfare(horizon)` equals
  `reward_ss * sum_t discount_rate**t`; `consumption_equivalent(w, w0, horizon)`
  inverts the utility so that scaling the baseline consumption path by
  `1 + x` gives `w`. Both depend only on the period utility.
- **Optional members.** `reset(rng, init_range)` and `step(rng, state, action)`
  are gym-style wrappers used by `smoke.py`. `get_aggregates(policies, states)`
  (policies first, the DEQN order) returns log deviations of the main
  aggregates along a simulation for plots and moment tables.
- **Precision.** Take `precision` and `double_precision` in the constructor and
  build every array with `self.precision`; tests that need float64 also set
  `jax_config.update("jax_enable_x64", True)`.

`RbcMultiSector` is a thin wrapper: it selects a model class from
`DEQN.econ_models.RBC` (`Model`, `ProjectedIrreversibleModel`, or
`ExactKinkModel`) and delegates every attribute it does not define to
`self.econ`. The policy is a saving-rate logit: zero network output is the
deterministic steady-state saving rate, mapped into
`(saving_rate_min, saving_rate_max)` by a sigmoid so consumption and investment
stay positive.

`TimeToBuildRbc` is the planner problem of Kydland and Prescott (1982) under
full information, self-contained in `environments/time_to_build.py` with their
Table I calibration as defaults: `J=4` build stages with equal shares,
non-time-separable leisure, inventories as a factor of production, and a
persistent plus a transitory productivity component. Spending on projects
already under way is committed before the period starts; the actions are the
share of free resources carried as inventories, the saving rate out of the
rest (new starts versus consumption), and hours, each through a bounded
sigmoid whose zero is the steady state, so the resource constraint holds and
every flow is positive by construction. `inventories`, `variable_labor`,
`n_stages`, and `transitory_shock_sd` switch features off; with `n_stages=1`,
fixed labor, and no inventories the environment has the same log-linear policy
as `RbcMultiSector` (`tests/test_time_to_build.py`). The steady state is
closed-form, and the LQ linearization's FOC residual at the steady state is the
regression test for it. Kydland and Prescott's noisy productivity indicator
and within-period two-stage timing are not implemented.

To add an environment, implement the table above and run the checker:

```python
from APG.environments import check_environment
check_environment(MyEnv())
```

It raises `AssertionError` naming the first violated property: shapes and
scalars are consistent, the steady-state action is a fixed point under a zero
shock, reward and next state stay finite for extreme actions, reward and
transition are differentiable in the action, `terminal_value` is differentiable
in the state, and the welfare helpers agree with the steady-state reward. The
same checker runs in `tests/test_environment_contract.py` on every RBC variant
and in `tests/test_time_to_build.py`. The checker does not verify that the
steady state is optimal or that the coordinate conventions above hold; for
that, run `APG.loglinear.solve.linearize_environment(env)` in float64 and
assert that `foc_residual` is zero. A wrong steady state, a missing
first-order condition, or an action map whose zero is not the steady state all
show up there.

A useful order of work, from `TimeToBuildRbc`:

1. Derive the deterministic steady state in closed form or with a root finder
   and store the levels; compute the action offsets (`policies_ss`) from them.
2. Write `allocation(state, action)` returning every period flow, and build
   `training_reward` and `transition` on top of it; test the resource
   constraint with equality on random states and actions.
3. Run `check_environment`, then `linearize_environment` for the FOC residual.
4. Pass the environment to `create_epoch_train_fn`, `create_eval_fn`, and
   `run_experiment`; `train_time_to_build.py` is a template that uses only the
   contract. Its `train(config)` returns the run result together with the
   environment, the network, and the LQ solution, and its config separates
   `loglinear_baseline` (residual network around `C`) from
   `loglinear_normalize` (scale inputs and outputs by the LQ stationary
   standard deviations), so the two effects can be measured separately.
5. Evaluate the trained policy with `APG.algorithm.create_welfare_fn`, which
   rolls out any `policy_fn(obs) -> action` with common random numbers; the
   same `rng` gives every policy the same initial levels and shock paths, so
   welfare differences can be converted with `consumption_equivalent`. The
   welfare gaps of interest are often ~1e-5 over 1024-period sums, which is
   near float32 resolution; evaluate in float64.

The Euler and welfare evaluations in `train.py` are RBC-specific and take
`env.econ`; a new environment has no Euler diagnostic unless it writes one.
Everything under `constrained/` reads RBC attributes (`alpha`, `delta`,
`phi`, `i_min_frac`, `normalized_investment_slack`, `constraint_state_grid`)
and is not model-agnostic.

## Constrained Variants

`APG/constrained/train.py` adds an investment floor `I >= i_min_frac * I_ss`
and selects how it enters training with `investment_constraint_mode`:

| Mode | Model | Treatment |
| --- | --- | --- |
| `project` | `ProjectedIrreversibleModel` | clip investment to the floor inside the step |
| `penalty` | `Model` | subtract a utility-scaled shortfall penalty from the reward |
| `learned_multiplier` | `Model` | primal-dual: `MultiplierNet` or `GridMultiplierNet`, augmented Lagrangian, warmup |
| `exact_kink` | `ExactKinkModel` | explicit kink in the policy map, Euler regularizer, optional critic |

`--irreversible` applies `IRREVERSIBLE_RBC_OVERLAY` (`i_min_frac=0.975`,
`phi=0`). The remaining keys in that file's `config` are mode-specific; the
`main()` validation block lists which combinations are allowed.

## Convergence Diagnostics

`algorithm/eval.py` has `create_convergence_eval_fn`, a many-rollout gradient
diagnostic. It averages the episode gradient over `diag_n_epis` rollouts and
reports `actor_grad_norm`, `critic_grad_norm`, `total_grad_norm`, their RMS
counterparts, `max_abs_grad`, and the critic's `value_loss` and
`value_accuracy`. Actor leaves are those under the `policy` sub-module of
`ActorCritic`; with a plain `PolicyNet` all leaves are actor leaves. The actor
gradient is the first-order convergence check; the critic numbers say whether
the value head fits its targets, not whether the policy is optimal.

## Tests

`APG/tests/` covers what APG adds: the environment contract on every RBC
variant (`test_environment_contract.py`), agreement between the APG and DEQN
rollouts plus antithetic and terminal-value behaviour (`test_rollout.py`), the
in-house log-linear policy (`test_loglinear.py`), the primal-dual multiplier's
direction and detachment (`test_constrained_apg.py`), the exact-kink
trainer (`test_exact_kink.py`), and the time-to-build economy's contract,
steady state, resource accounting, build timing, and RBC nesting
(`test_time_to_build.py`). Invariants of the RBC models themselves
(steady state, Euler and KKT residuals, welfare evaluation, kink policy map)
live in `DEQN/tests/test_rbc_model.py` and
`DEQN/tests/test_rbc_exact_kink_model.py`.

## Outputs

`run_experiment` writes to `APG/results/<run_name>/`: an Orbax checkpoint
(`max_to_keep=1`), `params.msgpack` (Flax serialization of the final params),
`results.json` (config plus loss, gradient, learning-rate, and welfare
histories), and PNG plots when `generate_plots` is on.

## Dependencies

JAX, Flax, Optax, Orbax-checkpoint, Matplotlib; see `requirements-dev.txt`.
