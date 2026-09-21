"""Naive-projected irreversible-investment benchmark.

Train or load an unconstrained saving-rate policy, then evaluate
`I_naive(x) = max(I_unc(x), I_min)` under the constrained law of motion.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random


class NaiveProjectedMetrics(NamedTuple):
    welfare: jax.Array
    welfare_se: jax.Array
    welfare_naive: jax.Array
    welfare_naive_se: jax.Array
    welfare_fixed_s: jax.Array
    ce_trained_vs_fixed: jax.Array
    ce_naive_vs_fixed: jax.Array
    ce_trained_vs_naive: jax.Array
    binding_frequency: jax.Array
    clipping_mass: jax.Array
    mean_unc_investment_over_iss: jax.Array
    mean_naive_investment_over_iss: jax.Array
    K_rel: jax.Array
    std_i: jax.Array


def _mean_and_se(samples):
    n = samples.shape[0]
    mean = jnp.mean(samples, axis=0)
    se = jnp.std(samples, axis=0, ddof=1) / jnp.sqrt(n)
    return mean, se


def create_naive_projected_eval_fn(
    unconstrained_model,
    constrained_model,
    unconstrained_policy_fn: Callable,
    trained_policy_fn: Callable | None = None,
    horizon: int = 256,
    n_epis: int = 1024,
    init_range: int = 0,
    policy_beta=None,
    hard_floor=None,
):
    """Paired evaluation of a trained policy against the naive projection.

    `unconstrained_policy_fn` is simulated with `I = max(I_unc, I_min)` on
    `constrained_model`. If `trained_policy_fn` is omitted, the trained-policy
    slots copy the naive policy so the CE gap is zero.
    """
    zero_action = constrained_model.deterministic_steady_state_action()
    discount_rate = constrained_model.discount_rate

    def unconstrained_investment(model, policy, obs):
        K, a = model._capital_and_productivity(obs)
        output = model.production(K, a)
        investment, _, _ = model.allocation_from_policy(
            policy, output, False, policy_beta, hard_floor
        )
        return investment, output

    def naive_step(obs, period_rng, get_unc_policy):
        investment_unc, output = unconstrained_investment(
            unconstrained_model, get_unc_policy(obs), obs
        )
        investment = constrained_model.naive_projected_investment(investment_unc)
        consumption = output - investment
        shock = constrained_model.sample_shock(period_rng)
        K, a = constrained_model._capital_and_productivity(obs)
        a_next = constrained_model.rho * a + constrained_model.shock_sd * shock
        K_next = constrained_model.next_capital(K, investment)
        obs_next = constrained_model._normalize_state(K_next, a_next)
        reward = constrained_model.period_utility(constrained_model.aggregate_consumption(consumption))
        clip = jnp.maximum(constrained_model.I_min - investment_unc, 0) / constrained_model.I_ss
        bind = constrained_model.constraint_binds(investment).astype(constrained_model.precision)
        return obs_next, reward, clip, bind, investment, investment_unc

    def trained_step(obs, period_rng, get_trained_policy):
        policy = get_trained_policy(obs)
        reward = jnp.reshape(
            constrained_model.reward(obs, policy, None, policy_beta, hard_floor),
            (),
        )
        shock = constrained_model.sample_shock(period_rng)
        obs_next = constrained_model.step(
            obs, policy, shock, None, policy_beta, hard_floor
        )
        K, a = constrained_model._capital_and_productivity(obs)
        output = constrained_model.production(K, a)
        investment, _, _ = constrained_model.allocation_from_policy(
            policy, output, None, policy_beta, hard_floor
        )
        return obs_next, reward, investment

    def fixed_step(obs, period_rng):
        reward = jnp.reshape(
            constrained_model.reward(obs, zero_action, project_investment=True),
            (),
        )
        shock = constrained_model.sample_shock(period_rng)
        obs_next = constrained_model.step(obs, zero_action, shock, project_investment=True)
        return obs_next, reward

    def rollout_naive(obs0, period_rngs, unc_policy):
        def period(carry, period_rng):
            obs, welfare, discount, clip_sum, bind_sum, i_sum, i_unc_sum, k_sum = carry
            obs_next, reward, clip, bind, investment, investment_unc = naive_step(
                obs, period_rng, unc_policy
            )
            K, _ = constrained_model._capital_and_productivity(obs)
            return (
                obs_next,
                welfare + discount * reward,
                discount * discount_rate,
                clip_sum + jnp.mean(clip),
                bind_sum + jnp.mean(bind),
                i_sum + jnp.mean(investment / constrained_model.I_ss),
                i_unc_sum + jnp.mean(investment_unc / constrained_model.I_ss),
                k_sum + jnp.mean(K / constrained_model.K_ss),
            ), jnp.log(jnp.mean(investment))

        init = (
            obs0,
            jnp.zeros((), dtype=constrained_model.precision),
            jnp.ones((), dtype=constrained_model.precision),
            jnp.zeros((), dtype=constrained_model.precision),
            jnp.zeros((), dtype=constrained_model.precision),
            jnp.zeros((), dtype=constrained_model.precision),
            jnp.zeros((), dtype=constrained_model.precision),
            jnp.zeros((), dtype=constrained_model.precision),
        )
        (
            _,
            welfare,
            _,
            clip_sum,
            bind_sum,
            i_sum,
            i_unc_sum,
            k_sum,
        ), log_i = lax.scan(period, init, period_rngs)
        return (
            welfare,
            clip_sum / horizon,
            bind_sum / horizon,
            i_sum / horizon,
            i_unc_sum / horizon,
            k_sum / horizon,
            jnp.std(log_i),
        )

    def rollout_trained(obs0, period_rngs, trained_policy):
        def period(carry, period_rng):
            obs, welfare, discount = carry
            obs_next, reward, _ = trained_step(obs, period_rng, trained_policy)
            return (obs_next, welfare + discount * reward, discount * discount_rate), None

        (_, welfare, _), _ = lax.scan(
            period,
            (
                obs0,
                jnp.zeros((), dtype=constrained_model.precision),
                jnp.ones((), dtype=constrained_model.precision),
            ),
            period_rngs,
        )
        return welfare

    def rollout_fixed(obs0, period_rngs):
        def period(carry, period_rng):
            obs, welfare, discount = carry
            obs_next, reward = fixed_step(obs, period_rng)
            return (obs_next, welfare + discount * reward, discount * discount_rate), None

        (_, welfare, _), _ = lax.scan(
            period,
            (
                obs0,
                jnp.zeros((), dtype=constrained_model.precision),
                jnp.ones((), dtype=constrained_model.precision),
            ),
            period_rngs,
        )
        return welfare

    def episode(unconstrained_params, trained_params, key):
        init_rng, periods_rng = random.split(key)
        period_rngs = random.split(periods_rng, horizon)
        obs0 = constrained_model.initial_state(init_rng, init_range=init_range)
        unc_policy_fn = lambda obs: unconstrained_policy_fn(unconstrained_params, obs)
        naive = rollout_naive(obs0, period_rngs, unc_policy_fn)
        if trained_policy_fn is None:
            trained_welfare = naive[0]
        else:
            trained_welfare = rollout_trained(
                obs0,
                period_rngs,
                lambda obs: trained_policy_fn(trained_params, obs),
            )
        fixed_welfare = rollout_fixed(obs0, period_rngs)
        return (trained_welfare, fixed_welfare, *naive)

    @jax.jit
    def evaluate(unconstrained_params, trained_params, rng):
        keys = random.split(rng, n_epis)
        (
            trained_welfare,
            fixed_welfare,
            naive_welfare,
            clipping_mass,
            binding,
            i_naive,
            i_unc,
            k_rel,
            std_i,
        ) = jax.vmap(episode, in_axes=(None, None, 0))(
            unconstrained_params, trained_params, keys
        )
        trained_mean, trained_se = _mean_and_se(trained_welfare)
        naive_mean, naive_se = _mean_and_se(naive_welfare)
        fixed_mean, _ = _mean_and_se(fixed_welfare)
        return NaiveProjectedMetrics(
            welfare=trained_mean,
            welfare_se=trained_se,
            welfare_naive=naive_mean,
            welfare_naive_se=naive_se,
            welfare_fixed_s=fixed_mean,
            ce_trained_vs_fixed=constrained_model.consumption_equivalent(
                trained_mean, fixed_mean, horizon
            ),
            ce_naive_vs_fixed=constrained_model.consumption_equivalent(
                naive_mean, fixed_mean, horizon
            ),
            ce_trained_vs_naive=constrained_model.consumption_equivalent(
                trained_mean, naive_mean, horizon
            ),
            binding_frequency=jnp.mean(binding),
            clipping_mass=jnp.mean(clipping_mass),
            mean_unc_investment_over_iss=jnp.mean(i_unc),
            mean_naive_investment_over_iss=jnp.mean(i_naive),
            K_rel=jnp.mean(k_rel),
            std_i=jnp.mean(std_i),
        )

    return evaluate


def naive_projected_metrics_to_dict(metrics: NaiveProjectedMetrics):
    return {
        "welfare": float(metrics.welfare),
        "welfare_se": float(metrics.welfare_se),
        "welfare_naive": float(metrics.welfare_naive),
        "welfare_naive_se": float(metrics.welfare_naive_se),
        "welfare_fixed_s": float(metrics.welfare_fixed_s),
        "ce_trained_vs_fixed": float(metrics.ce_trained_vs_fixed),
        "ce_naive_vs_fixed": float(metrics.ce_naive_vs_fixed),
        "ce_trained_vs_naive": float(metrics.ce_trained_vs_naive),
        "binding_frequency": float(metrics.binding_frequency),
        "clipping_mass": float(metrics.clipping_mass),
        "mean_unc_investment_over_iss": float(metrics.mean_unc_investment_over_iss),
        "mean_naive_investment_over_iss": float(metrics.mean_naive_investment_over_iss),
        "K_rel": float(metrics.K_rel),
        "std_i": float(metrics.std_i),
    }


def gate_decision(metrics: NaiveProjectedMetrics, solver_ce_accuracy=2e-4, min_ce_gap=None):
    """Stage 0 go/no-go. Recalibrate if the kink is economically negligible."""
    clipping = float(metrics.clipping_mass)
    ce_gap = abs(float(metrics.ce_trained_vs_naive))
    required_gap = 10 * solver_ce_accuracy if min_ce_gap is None else min_ce_gap
    recalibrate = clipping < 0.01 and ce_gap < required_gap
    return {
        "clipping_mass": clipping,
        "ce_trained_vs_naive": ce_gap,
        "required_ce_gap": required_gap,
        "binding_frequency": float(metrics.binding_frequency),
        "recalibrate": recalibrate,
        "reason": (
            "clipping mass and CE gap are both too small"
            if recalibrate
            else "kink is economically material enough to proceed"
        ),
    }
