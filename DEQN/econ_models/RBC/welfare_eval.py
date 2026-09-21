"""Finite-horizon consumption-equivalent welfare for the shared RBC model.

This is the APG diagnostic: discounted CES/CRRA utility, no terminal bootstrap,
CE gain vs the deterministic steady state and vs a fixed steady-state saving rate.
It is not the RbcProdNet Lucas-cost pipeline in DEQN.analysis.welfare.

Volatilities are mean across episodes of the within-episode std of log C, Y, I.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random


class WelfareMetrics(NamedTuple):
    welfare: jax.Array
    welfare_se: jax.Array
    welfare_ss: jax.Array
    welfare_fixed_s: jax.Array
    welfare_fixed_s_se: jax.Array
    ce_vs_no_shocks_ss: jax.Array
    ce_vs_fixed_saving_rate: jax.Array
    K_rel: jax.Array
    s_mean: jax.Array
    std_c: jax.Array
    std_y: jax.Array
    std_i: jax.Array
    i_bind_frac: jax.Array
    i_violation_frac: jax.Array
    i_violation_mean: jax.Array
    i_violation_max: jax.Array


def _mean_and_se(samples):
    n = samples.shape[0]
    mean = jnp.mean(samples, axis=0)
    se = jnp.std(samples, axis=0, ddof=1) / jnp.sqrt(n)
    return mean, se


def _log_mean(x):
    return jnp.log(jnp.mean(x))


def create_welfare_eval_fn(
    model,
    policy_fn: Callable,
    horizon: int,
    n_epis: int = 1024,
    init_range: int = 0,
    init_range_a=None,
    policy_beta=None,
    hard_floor=None,
):
    """Build a welfare eval that maps (params, rng) to WelfareMetrics.

    `policy_fn(params, obs)` must return the action/policy vector.
    Rollouts start at `init_range` (0 = deterministic SS). The learned policy
    and the fixed-`s_ss` baseline share the same initial state and shocks. The
    baseline remains projected when the learned policy is evaluated raw.
    """
    zero_action = model.deterministic_steady_state_action()
    welfare_ss = model.deterministic_steady_state_welfare(horizon)
    discount_rate = model.discount_rate

    def rollout(obs0, period_rngs, get_policy, project_investment=None):
        def period_step(carry, period_rng):
            obs, welfare, discount, k_sum, s_sum, bind_sum, violation_sum, violation_max = carry
            policy = get_policy(obs)
            reward = jnp.reshape(
                model.reward(obs, policy, project_investment, policy_beta, hard_floor),
                (),
            )
            K, a = model._capital_and_productivity(obs)
            output = model.production(K, a)
            investment, consumption, saving_rate = model.allocation_from_policy(
                policy, output, project_investment, policy_beta, hard_floor
            )
            shock = model.sample_shock(period_rng)
            obs_next = model.step(obs, policy, shock, project_investment, policy_beta, hard_floor)
            k_rel = jnp.mean(K / model.K_ss)
            s_mean = jnp.mean(saving_rate)
            bind = jnp.mean(model.constraint_binds(investment).astype(model.precision))
            shortfall_rel = model.investment_shortfall(investment) / model.I_ss
            violation = jnp.mean((shortfall_rel > 0).astype(model.precision))
            mean_shortfall = jnp.mean(shortfall_rel)
            max_shortfall = jnp.max(shortfall_rel)
            return (
                obs_next,
                welfare + discount * reward,
                discount * discount_rate,
                k_sum + k_rel,
                s_sum + s_mean,
                bind_sum + bind,
                violation_sum + jnp.array([violation, mean_shortfall]),
                jnp.maximum(violation_max, max_shortfall),
            ), (_log_mean(consumption), _log_mean(output), _log_mean(investment))

        init_carry = (
            obs0,
            jnp.zeros((), dtype=model.precision),
            jnp.ones((), dtype=model.precision),
            jnp.zeros((), dtype=model.precision),
            jnp.zeros((), dtype=model.precision),
            jnp.zeros((), dtype=model.precision),
            jnp.zeros(2, dtype=model.precision),
            jnp.zeros((), dtype=model.precision),
        )
        (_, welfare, _, k_sum, s_sum, bind_sum, violation_sum, violation_max), (log_c, log_y, log_i) = lax.scan(
            period_step, init_carry, period_rngs
        )
        return (
            welfare,
            k_sum / horizon,
            s_sum / horizon,
            bind_sum / horizon,
            violation_sum[0] / horizon,
            violation_sum[1] / horizon,
            violation_max,
            jnp.std(log_c),
            jnp.std(log_y),
            jnp.std(log_i),
        )

    def episode_metrics(params, key):
        init_rng, periods_rng = random.split(key)
        period_rngs = random.split(periods_rng, horizon)
        obs0 = model.initial_state(
            init_rng, init_range=init_range, init_range_a=init_range_a, mode="box"
        )
        welfare_pi, k_rel, s_mean, i_bind_frac, violation_frac, violation_mean, violation_max, std_c, std_y, std_i = rollout(
            obs0, period_rngs, lambda obs: policy_fn(params, obs)
        )
        welfare_fixed, _, _, _, _, _, _, _, _, _ = rollout(
            obs0,
            period_rngs,
            lambda obs: zero_action,
            project_investment=True,
        )
        return (
            welfare_pi,
            welfare_fixed,
            k_rel,
            s_mean,
            i_bind_frac,
            violation_frac,
            violation_mean,
            violation_max,
            std_c,
            std_y,
            std_i,
        )

    @jax.jit
    def welfare_eval_fn(params, rng):
        keys = random.split(rng, n_epis)
        (
            welfare_pi,
            welfare_fixed,
            k_rel,
            s_mean,
            i_bind_frac,
            violation_frac,
            violation_mean,
            violation_max,
            std_c,
            std_y,
            std_i,
        ) = jax.vmap(
            episode_metrics, in_axes=(None, 0)
        )(params, keys)
        welfare_mean, welfare_se = _mean_and_se(welfare_pi)
        fixed_mean, fixed_se = _mean_and_se(welfare_fixed)
        return WelfareMetrics(
            welfare=welfare_mean,
            welfare_se=welfare_se,
            welfare_ss=welfare_ss,
            welfare_fixed_s=fixed_mean,
            welfare_fixed_s_se=fixed_se,
            ce_vs_no_shocks_ss=model.consumption_equivalent(welfare_mean, welfare_ss, horizon),
            ce_vs_fixed_saving_rate=model.consumption_equivalent(welfare_mean, fixed_mean, horizon),
            K_rel=jnp.mean(k_rel),
            s_mean=jnp.mean(s_mean),
            std_c=jnp.mean(std_c),
            std_y=jnp.mean(std_y),
            std_i=jnp.mean(std_i),
            i_bind_frac=jnp.mean(i_bind_frac),
            i_violation_frac=jnp.mean(violation_frac),
            i_violation_mean=jnp.mean(violation_mean),
            i_violation_max=jnp.max(violation_max),
        )

    return welfare_eval_fn


def print_welfare_metrics(metrics: WelfareMetrics, prefix="  Welfare"):
    print(
        f"{prefix}: W={float(metrics.welfare):.4f} "
        f"(se={float(metrics.welfare_se):.4f}) "
        f"CE vs SS={100 * float(metrics.ce_vs_no_shocks_ss):+.4f}% "
        f"CE vs s_ss={100 * float(metrics.ce_vs_fixed_saving_rate):+.4f}% "
        f"K/Kss={float(metrics.K_rel):.4f} s={float(metrics.s_mean):.4f} "
        f"bind={100 * float(metrics.i_bind_frac):.1f}% "
        f"violate={100 * float(metrics.i_violation_frac):.2f}%",
        flush=True,
    )
    print(
        f"{prefix} vols: std(log C)={float(metrics.std_c):.4f} "
        f"std(log Y)={float(metrics.std_y):.4f} "
        f"std(log I)={float(metrics.std_i):.4f}",
        flush=True,
    )


def welfare_metrics_to_dict(metrics: WelfareMetrics):
    return {
        "welfare": float(metrics.welfare),
        "welfare_se": float(metrics.welfare_se),
        "welfare_ss": float(metrics.welfare_ss),
        "welfare_fixed_s": float(metrics.welfare_fixed_s),
        "welfare_fixed_s_se": float(metrics.welfare_fixed_s_se),
        "ce_vs_no_shocks_ss": float(metrics.ce_vs_no_shocks_ss),
        "ce_vs_fixed_saving_rate": float(metrics.ce_vs_fixed_saving_rate),
        "K_rel": float(metrics.K_rel),
        "s_mean": float(metrics.s_mean),
        "std_c": float(metrics.std_c),
        "std_y": float(metrics.std_y),
        "std_i": float(metrics.std_i),
        "i_bind_frac": float(metrics.i_bind_frac),
        "i_violation_frac": float(metrics.i_violation_frac),
        "i_violation_mean": float(metrics.i_violation_mean),
        "i_violation_max": float(metrics.i_violation_max),
    }
