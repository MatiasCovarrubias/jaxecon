#!/usr/bin/env python3
"""
Training script for the RBC model with DEQN.

Usage:
    LOCAL:
        python -m DEQN.econ_models.RBC.train
        # or
        python DEQN/econ_models/RBC/train.py

    COLAB:
        Copy contents into a cell and run.
"""

import json
import os
import sys

# ============================================================================
# ENVIRONMENT DETECTION AND SETUP
# ============================================================================

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Environment: {'Google Colab' if IN_COLAB else 'Local'}")

if IN_COLAB:
    import subprocess

    def _colab_package_stack_is_usable() -> bool:
        try:
            import jax
            import numpy
            import scipy
            import scipy.io  # noqa: F401
        except Exception as exc:
            print(f"Package stack check failed in current kernel: {exc!r}")
            return False
        print("Package stack OK: " f"numpy={numpy.__version__} scipy={scipy.__version__} jax={jax.__version__}")
        return True

    _COLAB_REPAIR_MARKER = "/content/.jaxecon_colab_numpy_repair_attempted"
    if _colab_package_stack_is_usable():
        if os.path.exists(_COLAB_REPAIR_MARKER):
            os.remove(_COLAB_REPAIR_MARKER)
    else:
        if os.path.exists(_COLAB_REPAIR_MARKER):
            raise RuntimeError(
                "The Colab NumPy/SciPy/JAX stack is still inconsistent after a repair restart. "
                "Use Runtime > Disconnect and delete runtime, then rerun the notebook."
            )
        print("Installing pinned NumPy/SciPy/JAX stack, then restarting Colab.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "numpy==2.0.2",
                "scipy==1.15.3",
                "jax[cuda12]",
            ],
            check=True,
        )
        with open(_COLAB_REPAIR_MARKER, "w", encoding="utf-8") as marker:
            marker.write("numpy==2.0.2 scipy==1.15.3 jax[cuda12]\n")
        print("Package stack repaired. Restarting Colab runtime; rerun this cell after reconnecting.")
        os.kill(os.getpid(), 9)

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"], check=True)

    sys.path.insert(0, "/content/jaxecon")
    repo_root = "/content/jaxecon"
    base_dir = "/content/drive/MyDrive/Jaxecon/DEQN/econ_models"

else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    base_dir = os.path.join(repo_root, "DEQN", "econ_models")

# ============================================================================
# IMPORTS
# ============================================================================

import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.algorithm.epoch_train import create_epoch_train_fn  # noqa: E402
from DEQN.algorithm.eval import create_eval_fn  # noqa: E402
from DEQN.algorithm.simulation import create_episode_simul_fn  # noqa: E402
from DEQN.econ_models.RBC.euler_eval import euler_metrics_to_dict  # noqa: E402
from DEQN.econ_models.RBC.irreversible_model import (  # noqa: E402
    IrreversibleModel,
)
from DEQN.econ_models.RBC.model import Model  # noqa: E402
from DEQN.econ_models.RBC.train_shared import (  # noqa: E402
    DEQN_EVAL_MC_DRAWS,
    DEQN_LEARNING_RATE,
    DEQN_MC_DRAWS,
    SHARED_RBC_TRAIN,
    create_policy_train_state,
    init_policy_params,
    maybe_save_policy_snapshot,
    resolve_policy_snapshot_epochs,
    save_policy_params,
    split_shared_rbc_rngs,
    shared_model_kwargs,
    with_derived_counts,
    with_irreversible,
)
from DEQN.econ_models.RBC.welfare_eval import (  # noqa: E402
    create_welfare_eval_fn,
    print_welfare_metrics,
    welfare_metrics_to_dict,
)
from DEQN.neural_nets.neural_nets import PolicyNet  # noqa: E402
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet as NeuralNet_loglinear  # noqa: E402

jax_config.update("jax_debug_nans", True)

# ============================================================================
# CONFIGURATION
# ============================================================================

config = with_irreversible(
    with_derived_counts(
        {
            **SHARED_RBC_TRAIN,
            "exper_name": "rbc_baseline",
            "working_dir": os.path.join(base_dir, "RBC", "results"),
            "learning_rate": DEQN_LEARNING_RATE,
            "mc_draws": DEQN_MC_DRAWS,
            "config_eval": {
                "periods_per_epis": SHARED_RBC_TRAIN["eval_periods_per_epis"],
                "mc_draws": DEQN_EVAL_MC_DRAWS,
                "simul_vol_scale": SHARED_RBC_TRAIN["simul_vol_scale"],
                "eval_n_epis": SHARED_RBC_TRAIN["eval_n_epis"],
                "init_range": SHARED_RBC_TRAIN["init_range"],
                "init_range_a": SHARED_RBC_TRAIN["init_range_a"],
            },
        }
    ),
    enabled="--irreversible" in sys.argv,
)

# ============================================================================
# MAIN
# ============================================================================


def main():
    from time import time

    import jax

    print(f"\n{'='*60}")
    print("DEQN Training: RBC Model")
    print(f"{'='*60}\n")

    # Precision setup
    precision = jnp.float64 if config["double_precision"] else jnp.float32
    if config["double_precision"]:
        jax_config.update("jax_enable_x64", True)

    # Create economic model
    print("Creating economic model...")
    model_class = (
        IrreversibleModel if config["i_min_frac"] > 0 else Model
    )
    econ_model = model_class(
        **shared_model_kwargs(config),
        precision=precision,
        double_precision=config["double_precision"],
    )
    print(f"  State dim: {econ_model.dim_states}")
    print(f"  Policy dim: {econ_model.dim_policies}")
    print(f"  beta: {float(jnp.squeeze(econ_model.beta)):.2f}")
    print(f"  alpha: {float(jnp.squeeze(econ_model.alpha)):.2f}")
    print(f"  delta: {float(jnp.squeeze(econ_model.delta)):.2f}")
    print(f"  shock_sd: {float(jnp.squeeze(econ_model.shock_sd)):.2f}")
    print(f"  rho: {float(jnp.squeeze(econ_model.rho)):.2f}")
    print(f"  phi: {float(jnp.squeeze(econ_model.phi)):.2f}")
    print(f"  i_min_frac: {float(jnp.squeeze(econ_model.i_min_frac)):.3f}")
    print(f"  eps_c (IES): {float(jnp.squeeze(econ_model.eps_c)):.2f}")
    print(f"  risk aversion: {float(jnp.squeeze(econ_model.risk_aversion)):.2f}")
    print(f"  K_ss (log): {float(jnp.squeeze(econ_model.k_ss)):.4f}")
    print(f"  s_ss: {float(jnp.squeeze(econ_model.s_ss)):.4f}")

    from APG.loglinear.design import needs_lq_objects

    if needs_lq_objects(config):
        from APG.environments import RbcMultiSector
        from APG.loglinear.design import apply_lq_design, design_from_config

        if "lq_A" in config:
            apply_lq_design(None, econ_model, config, design=design_from_config(config))
        else:
            model_kwargs = shared_model_kwargs(config)
            n_sectors = model_kwargs.pop("n_sectors")
            design_env = RbcMultiSector(
                N=n_sectors,
                **model_kwargs,
                project_investment=True,
                policy_map="saving_rate",
                double_precision=config["double_precision"],
                precision=precision,
            )
            apply_lq_design(design_env, econ_model, config)

    # Create neural network
    print("\nCreating neural network...")
    if config.get("loglinear_baseline"):
        neural_net = NeuralNet_loglinear(
            features=config["layers"],
            C=jnp.asarray(config["loglinear_C"], dtype=precision),
            states_sd=jnp.asarray(econ_model.state_sd, dtype=precision),
            policies_sd=jnp.asarray(econ_model.policies_sd, dtype=precision),
            param_dtype=precision,
        )
        print(f"  Architecture: LQ residual {config['layers']} -> {econ_model.dim_policies}")
    else:
        neural_net = PolicyNet(
            features=config["layers"],
            n_out=econ_model.dim_policies,
            precision=precision,
        )
        print(f"  Architecture: {config['layers']} -> {econ_model.dim_policies}")

    # Initialize
    streams = split_shared_rbc_rngs(config["seed"])
    rng, rng_init, rng_epoch, rng_eval, rng_welfare = (
        streams["leftover"],
        streams["init"],
        streams["epoch"],
        streams["eval"],
        streams["welfare"],
    )

    params = init_policy_params(neural_net, rng_init, econ_model.state_ss)
    train_state_obj, lr_schedule = create_policy_train_state(neural_net, params, config)

    # Create training and eval functions
    print("\nCompiling training functions...")
    epoch_train_fn = jax.jit(create_epoch_train_fn(econ_model, config))
    eval_fn = jax.jit(create_eval_fn(econ_model, config))
    welfare_eval_fn = create_welfare_eval_fn(
        econ_model,
        policy_fn=lambda params, obs: neural_net.apply(params, obs),
        horizon=config.get("welfare_history_horizon", config["welfare_horizon"]),
        n_epis=config["welfare_n_epis"],
        init_range=config["welfare_init_range"],
        init_range_a=config.get("welfare_init_range_a"),
    )

    # Warmup compilation
    t0 = time()
    _ = epoch_train_fn(train_state_obj, rng_epoch)
    _ = eval_fn(train_state_obj, rng_eval)
    _ = welfare_eval_fn(train_state_obj.params, rng_welfare)
    print(f"  Compilation time: {time() - t0:.2f}s")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} {'Loss':>12} {'Mean Acc':>12} {'Min Acc':>12} {'LR':>12}")
    print("-" * 60)

    t_start = time()
    losses, accuracies = [], []
    welfare_history = []
    final_welfare = None
    run_dir = os.path.join(config["working_dir"], config["exper_name"])
    os.makedirs(run_dir, exist_ok=True)
    snapshot_epochs = resolve_policy_snapshot_epochs(config)

    welfare_every = int(config.get("welfare_every_n_epochs") or 1)
    if welfare_every < 1:
        raise ValueError("welfare_every_n_epochs must be a positive integer")

    def record_eval(epoch, include_welfare=True):
        eval_metrics = eval_fn(train_state_obj, rng_eval)
        mean_loss = float(eval_metrics[0])
        mean_acc = float(eval_metrics[1])
        min_acc = float(eval_metrics[2])
        current_lr = float(lr_schedule(train_state_obj.step))
        losses.append(mean_loss)
        accuracies.append(mean_acc)
        print(f"{epoch:>6} {mean_loss:>12.6f} {mean_acc:>12.4f} {min_acc:>12.4f} {current_lr:>12.6f}")
        welfare = euler_metrics_to_dict((mean_loss, mean_acc, min_acc))
        if include_welfare:
            welfare_metrics = welfare_eval_fn(train_state_obj.params, rng_welfare)
            print_welfare_metrics(welfare_metrics)
            welfare.update(welfare_metrics_to_dict(welfare_metrics))
            welfare_history.append({"epoch": epoch, **welfare})
        return welfare

    final_welfare = record_eval(0)
    maybe_save_policy_snapshot(train_state_obj.params, run_dir, 0, snapshot_epochs)

    for epoch in range(1, config["n_epochs"] + 1):
        train_state_obj, rng_epoch, _epoch_metrics = epoch_train_fn(train_state_obj, rng_epoch)
        include_welfare = epoch == config["n_epochs"] or epoch % welfare_every == 0
        final_welfare = record_eval(epoch, include_welfare=include_welfare)
        maybe_save_policy_snapshot(train_state_obj.params, run_dir, epoch, snapshot_epochs)

    total_time = time() - t_start
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Final accuracy: {accuracies[-1]:.4f}")

    results = {
        "run_name": config["exper_name"],
        "config": {k: v for k, v in config.items() if not callable(v)},
        "losses": losses,
        "accuracies": accuracies,
        "time_train_seconds": total_time,
        "welfare": final_welfare,
        "welfare_history": welfare_history,
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    save_policy_params(train_state_obj.params, os.path.join(run_dir, "params.msgpack"))

    if config.get("post_training_analysis", True):
        print(f"\n{'='*60}")
        print("Running Simulation")
        print(f"{'='*60}")

        simul_fn = create_episode_simul_fn(
            econ_model,
            {
                "periods_per_epis": config["periods_per_epis"],
                "init_range": 0,
                "simul_vol_scale": 1.0,
                "antithetic_episodes": False,
            },
        )
        simul_obs = simul_fn(train_state_obj, rng)
        simul_policies = train_state_obj.apply_fn(train_state_obj.params, simul_obs)

        aggregates = econ_model.get_aggregates(simul_policies, simul_obs)

        print("\nSimulation Statistics (256 periods from steady state):")
        for name, values in aggregates.items():
            print(f"  {name}: mean={float(jnp.mean(values)):.4f}, std={float(jnp.std(values)):.4f}")

    return train_state_obj, losses, accuracies


if __name__ == "__main__":
    main()
