#!/usr/bin/env python3
"""Small APG smoke check for the two-sector RBC environment."""

import os
import sys

if __package__ in {None, ""}:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from APG.algorithm import create_epoch_train_fn, create_eval_fn
from APG.environments import RbcMultiSector
from APG.neural_nets import ActorCritic


def main():
    config = {
        "seed": 0,
        "n_sectors": 2,
        "steps_per_epoch": 1,
        "epis_per_step": 2,
        "periods_per_epis": 2,
        "eval_n_epis": 2,
        "eval_periods_per_epis": 2,
        "gae_lambda": 0.95,
    }

    env = RbcMultiSector(N=config["n_sectors"])
    neural_net = ActorCritic(
        actions_dim=env.action_dim,
        hidden_dims_actor=[4],
        hidden_dims_critic=[4],
    )

    rng = jax.random.PRNGKey(config["seed"])
    rng, rng_pol, rng_env, rng_epoch, rng_eval = jax.random.split(rng, 5)
    obs, _ = env.reset(rng_env)
    params = neural_net.init(rng_pol, obs)
    train_state_obj = train_state.TrainState.create(
        apply_fn=neural_net.apply,
        params=params,
        tx=optax.adam(0.001),
    )

    epoch_train_fn = create_epoch_train_fn(env, config)
    eval_fn = create_eval_fn(env, config)

    train_state_obj, _, epoch_metrics = epoch_train_fn(train_state_obj, rng_epoch)
    eval_metrics = eval_fn(train_state_obj, rng_eval)

    print("APG smoke ok")
    print(f"epoch_loss={float(jnp.mean(epoch_metrics[0][0])):.6f}")
    print(f"eval_loss={float(eval_metrics[0]):.6f}")


if __name__ == "__main__":
    main()
