"""Exact-kink APG: shared latent, complementary slackness by construction.

The actor outputs an unbounded latent `w`. Investment and the multiplier are

    I = I_min + (I_ss - I_min) * unit_excess(w)
    mu = kappa_mu * u'(C_ss) * softplus_beta(-w)

so the floor is hit exactly as beta -> inf and (I - I_min) * mu = 0. Training
keeps the APG welfare gradient and adds a pointwise Euler residual whose
target is a stop-gradient costate V_K from a differentiable rollout.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from DEQN.algorithm.simulation import sample_episode_shocks
from APG.algorithm.simulation import Transition


# 5-node Gauss-Hermite for E[f(Z)], Z ~ N(0, 1).
_GH_NODES = jnp.array(
    [-2.8569700138728056, -1.3556261799742658, 0.0, 1.3556261799742658, 2.8569700138728056]
)
_GH_WEIGHTS = jnp.array(
    [0.011257411327720689, 0.22207592251143874, 0.5333333333333332, 0.22207592251143874, 0.011257411327720689]
)


class ExactKinkMetrics(NamedTuple):
    actor_loss: jax.Array
    welfare_loss: jax.Array
    euler_loss: jax.Array
    latent_reg: jax.Array
    policy_beta: jax.Array
    safety_clip_frac: jax.Array
    bind_frac: jax.Array
    mean_latent: jax.Array
    corner_frac: jax.Array


def log_linear_beta(step, total_steps, beta_start, beta_end):
    start = jnp.asarray(beta_start)
    end = jnp.asarray(beta_end)
    t = jnp.clip(step.astype(start.dtype) / jnp.maximum(total_steps - 1, 1), 0, 1)
    return jnp.exp((1 - t) * jnp.log(start) + t * jnp.log(end))


def _policy_action(apply_fn, params, obs, use_terminal_value):
    output = apply_fn(params, obs)
    if use_terminal_value:
        return output[0]
    return output


def _maybe_checkpoint(fn, enabled):
    return jax.checkpoint(fn) if enabled else fn


def create_exact_kink_rollout(env, apply_fn, config):
    econ = env.econ
    periods = int(config.get("vk_horizon", config.get("periods_per_epis", 48)))
    use_terminal_value = bool(config.get("use_terminal_value", False))
    use_model_terminal_value = bool(
        config.get(
            "use_model_terminal_value",
            config.get("use_analytic_tail", False),
        )
    )
    terminal_value_horizon = int(
        config.get(
            "terminal_value_horizon",
            config.get("tail_periods", 512),
        )
    )
    rematerialize = bool(config.get("rematerialize_rollout", False)) or int(
        config.get("periods_per_epis", periods)
    ) >= 1024
    simul_vol_scale = config.get("simul_vol_scale", 1.0)
    if use_model_terminal_value and use_terminal_value:
        raise ValueError(
            "learned critic bootstrap and model terminal continuation cannot be combined"
        )
    if terminal_value_horizon < 1:
        raise ValueError("terminal_value_horizon must be positive")

    def rollout_from_state(params, state, shocks, policy_beta, hard_floor):
        def period_step(carry, shock):
            obs, returns, discount = carry
            action = _policy_action(apply_fn, params, obs, use_terminal_value)
            reward = econ.reward(obs, action, None, policy_beta, hard_floor)
            obs_next = econ.step(obs, action, shock, None, policy_beta, hard_floor)
            returns = returns + discount * reward
            discount = env.discount_rate * discount
            return (obs_next, returns, discount), None

        (last_obs, returns, discount), _ = lax.scan(
            _maybe_checkpoint(period_step, rematerialize),
            (
                state,
                jnp.zeros((), dtype=econ.precision),
                jnp.ones((), dtype=econ.precision),
            ),
            shocks,
        )
        if use_model_terminal_value:
            returns = returns + discount * env.terminal_value(
                last_obs,
                terminal_value_horizon,
            )
        elif use_terminal_value:
            _, last_val_notnorm = apply_fn(params, last_obs)
            returns = returns + discount * last_val_notnorm * env.value_ss
        return returns

    def rollout_return(capital, productivity, params, shocks, policy_beta, hard_floor):
        state = econ._normalize_state(jnp.reshape(capital, (econ.n_sectors,)), productivity)
        return rollout_from_state(params, state, shocks, policy_beta, hard_floor)

    value_capital = jax.grad(rollout_return, argnums=0)

    def simulate_episode(params, init_obs, shocks, policy_beta, hard_floor):
        def period_step(carry, shock):
            obs, returns, discount = carry
            action = _policy_action(apply_fn, params, obs, use_terminal_value)
            if use_terminal_value:
                _, value_notnorm = apply_fn(params, obs)
            else:
                value_notnorm = jnp.zeros((), dtype=econ.precision)
            reward = econ.reward(obs, action, None, policy_beta, hard_floor)
            obs_next = econ.step(obs, action, shock, None, policy_beta, hard_floor)
            transition = Transition(
                jnp.array(False),
                action,
                value_notnorm * env.value_ss,
                reward,
                obs_next,
                jnp.array([0.0]),
            )
            returns = returns + discount * reward
            discount = env.discount_rate * discount
            return (obs_next, returns, discount), transition

        (last_obs, returns, discount), trajectory = lax.scan(
            _maybe_checkpoint(period_step, rematerialize),
            (
                init_obs,
                jnp.zeros((), dtype=econ.precision),
                jnp.ones((), dtype=econ.precision),
            ),
            shocks,
        )
        if use_model_terminal_value:
            last_val = env.terminal_value(last_obs, terminal_value_horizon)
            returns = returns + discount * last_val
        elif use_terminal_value:
            _, last_val_notnorm = apply_fn(params, last_obs)
            last_val = last_val_notnorm * env.value_ss
            returns = returns + discount * last_val
        else:
            last_val = jnp.zeros_like(returns)
        return returns, trajectory, last_val

    return {
        "rollout_from_state": rollout_from_state,
        "rollout_return": rollout_return,
        "value_capital": value_capital,
        "simulate_episode": simulate_episode,
        "periods": periods,
        "simul_vol_scale": simul_vol_scale,
    }


def _state_grid(env, size, k_min, k_max, a_sd_min, a_sd_max):
    return env.constraint_state_grid(
        size=size,
        k_min=k_min,
        k_max=k_max,
        a_sd_min=a_sd_min,
        a_sd_max=a_sd_max,
    )


def create_exact_kink_objectives(env, apply_fn, config):
    econ = env.econ
    rollout = create_exact_kink_rollout(env, apply_fn, config)
    welfare_periods = int(config.get("periods_per_epis", 512))
    antithetic = bool(config.get("antithetic_episodes", False))
    init_range = config.get("init_range", 5)
    euler_weight = float(config.get("euler_weight", 1.0))
    latent_reg_weight = float(config.get("latent_reg_weight", 1e-4))
    latent_reg_threshold = float(config.get("latent_reg_threshold", 5.0))
    n_euler = int(config.get("n_euler_states", 256))
    grid_share = float(config.get("euler_grid_share", 0.5))
    occupancy_share = float(config.get("euler_occupancy_share", 0.2))
    corner_share = float(config.get("euler_corner_share", 0.3))
    corner_band = float(config.get("euler_corner_band", 0.5))
    grid_size = int(config.get("euler_grid_size", 64))
    k_min = float(config.get("constraint_grid_k_min", 0.85))
    k_max = float(config.get("constraint_grid_k_max", 1.20))
    a_sd_min = float(config.get("constraint_grid_a_sd_min", -4.0))
    a_sd_max = float(config.get("constraint_grid_a_sd_max", 4.0))
    use_terminal_value = bool(config.get("use_terminal_value", False))
    use_model_terminal_value = bool(
        config.get(
            "use_model_terminal_value",
            config.get("use_analytic_tail", False),
        )
    )
    terminal_value_horizon = int(
        config.get(
            "terminal_value_horizon",
            config.get("tail_periods", 512),
        )
    )
    gae_lambda = float(config.get("gae_lambda", 0.95))
    critic_coef = float(config.get("critic_coef", 1.0))
    welfare_scale = float(econ.discounted_period_weight(welfare_periods))
    if use_model_terminal_value:
        welfare_scale = welfare_scale + float(
            (econ.beta ** welfare_periods)
            * econ.discounted_period_weight(terminal_value_horizon)
        )
    n_grid = max(1, int(round(n_euler * grid_share)))
    n_occ = max(1, int(round(n_euler * occupancy_share)))
    n_corner = max(1, n_euler - n_grid - n_occ)
    gh_nodes = _GH_NODES.astype(econ.precision)
    gh_weights = _GH_WEIGHTS.astype(econ.precision)

    def occupancy_states(params, rng, policy_beta, hard_floor):
        init_obs = econ.initial_state(rng, init_range)
        shocks = sample_episode_shocks(econ, rng, welfare_periods, rollout["simul_vol_scale"])
        _, trajectory, _ = rollout["simulate_episode"](
            params, init_obs, shocks, policy_beta, hard_floor
        )
        return trajectory.obs

    def euler_rhs(params, capital, productivity, investment, shocks, policy_beta, hard_floor):
        k_next = econ.next_capital(capital, investment)

        def one_node(node):
            a_next = econ.rho * productivity + econ.shock_sd * node
            return rollout["value_capital"](
                k_next, a_next, params, shocks, policy_beta, hard_floor
            )

        values = jax.vmap(one_node)(gh_nodes)
        return econ.beta * jnp.tensordot(gh_weights, values, axes=(0, 0))

    def residual_at_state(params, state, shocks, policy_beta, hard_floor):
        action = _policy_action(apply_fn, params, state, use_terminal_value)
        K, a = econ._capital_and_productivity(state)
        output = econ.production(K, a)
        investment, consumption, _ = econ.allocation_from_policy(
            action, output, None, policy_beta, hard_floor
        )
        mu = econ.multiplier_from_latent(econ._policy_latent(action), policy_beta, hard_floor)
        rhs = euler_rhs(params, K, a, investment, shocks, policy_beta, hard_floor)
        rhs = lax.stop_gradient(rhs)
        utility_prime = econ.marginal_utility(consumption)
        residual = (utility_prime - mu - rhs) / utility_prime
        clip = econ.safety_clip_active(action, output, policy_beta, hard_floor)
        slack = econ.investment_slack(investment)
        return residual, action, slack, clip

    def mixed_states(params, rng, policy_beta, hard_floor):
        rng_grid, rng_occ, rng_corner, rng_occ_roll = random.split(rng, 4)
        grid = _state_grid(env, grid_size, k_min, k_max, a_sd_min, a_sd_max)
        grid_idx = random.choice(rng_grid, grid.shape[0], shape=(n_grid,), replace=True)
        grid_states = grid[grid_idx]
        occ = occupancy_states(params, rng_occ_roll, policy_beta, hard_floor)
        occ_idx = random.choice(rng_occ, occ.shape[0], shape=(n_occ,), replace=True)
        occ_states = occ[occ_idx]
        candidates = jnp.concatenate([grid, occ], axis=0)
        latents = jax.vmap(
            lambda state: econ._policy_latent(
                _policy_action(apply_fn, params, state, use_terminal_value)
            )
        )(candidates).reshape(-1)
        corner_score = -jnp.abs(latents)
        in_band = jnp.abs(latents) < corner_band
        score = jnp.where(in_band, corner_score + 10.0, corner_score)
        corner_idx = random.choice(
            rng_corner,
            candidates.shape[0],
            shape=(n_corner,),
            replace=True,
            p=jax.nn.softmax(score * 4.0),
        )
        return jnp.concatenate([grid_states, occ_states, candidates[corner_idx]], axis=0)

    def welfare_loss(params, rng, policy_beta, hard_floor):
        init_obs = econ.initial_state(rng, init_range)
        shocks = sample_episode_shocks(econ, rng, welfare_periods, rollout["simul_vol_scale"])
        returns, trajectory, last_val = rollout["simulate_episode"](
            params, init_obs, shocks, policy_beta, hard_floor
        )
        if antithetic:
            returns_minus, _, last_val_minus = rollout["simulate_episode"](
                params, init_obs, -shocks, policy_beta, hard_floor
            )
            returns = 0.5 * (returns + returns_minus)
            last_val = 0.5 * (last_val + last_val_minus)
        actor_loss = -returns / welfare_scale
        zero = jnp.zeros_like(actor_loss)
        if not use_terminal_value:
            return actor_loss, trajectory, zero, zero
        targets = _gae_targets(env, trajectory, last_val, gae_lambda)
        states = jnp.concatenate([init_obs[None, ...], trajectory.obs[:-1]], axis=0)

        def critic_at(obs):
            _, value_notnorm = apply_fn(params, obs)
            return value_notnorm * env.value_ss

        values_det = jax.vmap(critic_at)(lax.stop_gradient(states))
        value_loss = jnp.mean(jnp.square(values_det - targets))
        value_loss_perc = jnp.mean(
            (values_det - targets) / jnp.maximum(jnp.abs(targets), jnp.array(1e-6, dtype=econ.precision))
        )
        return actor_loss, trajectory, value_loss, value_loss_perc

    def objective(params, rng, policy_beta, hard_floor=False):
        rng_welfare, rng_states, rng_vk = random.split(rng, 3)
        welfare, trajectory, value_loss, value_loss_perc = welfare_loss(
            params, rng_welfare, policy_beta, hard_floor
        )
        states = mixed_states(params, rng_states, policy_beta, hard_floor)
        vk_shocks = sample_episode_shocks(
            econ, rng_vk, rollout["periods"], rollout["simul_vol_scale"]
        )
        residuals, actions, slacks, clips = jax.vmap(
            lambda state: residual_at_state(
                params, state, vk_shocks, policy_beta, hard_floor
            )
        )(states)
        euler_loss = jnp.mean(residuals**2)
        latent_reg = econ.latent_regularization(actions, latent_reg_threshold)
        loss = (
            welfare
            + euler_weight * euler_loss
            + latent_reg_weight * latent_reg
            + critic_coef * value_loss
        )
        metrics = ExactKinkMetrics(
            actor_loss=loss,
            welfare_loss=welfare,
            euler_loss=euler_loss,
            latent_reg=latent_reg,
            policy_beta=policy_beta,
            safety_clip_frac=jnp.mean(clips.astype(econ.precision)),
            bind_frac=jnp.mean((slacks <= 1e-3).astype(econ.precision)),
            mean_latent=jnp.mean(econ._policy_latent(actions)),
            corner_frac=jnp.mean((jnp.abs(econ._policy_latent(actions)) < corner_band).astype(econ.precision)),
        )
        return loss, (metrics, value_loss, value_loss_perc)

    return objective, rollout


def _gae_targets(env, trajectory, last_val, gae_lambda):
    def get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + env.discount_rate * next_value * (1 - transition.done) - transition.value
        gae = delta + env.discount_rate * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = lax.scan(
        get_advantages,
        (jnp.zeros_like(last_val), last_val),
        trajectory,
        reverse=True,
    )
    return lax.stop_gradient(advantages + trajectory.value)


def create_exact_kink_epoch_train_fn(env, apply_fn, config):
    objective, _ = create_exact_kink_objectives(env, apply_fn, config)
    total_steps = max(1, int(config["n_epochs"]) * int(config["steps_per_epoch"]))
    beta_start = float(config.get("policy_beta_start", 10.0))
    beta_end = float(config.get("policy_beta_end", 500.0))
    hard_floor = bool(config.get("hard_floor", False))

    def episode_train(train_state, epis_rng):
        policy_beta = log_linear_beta(train_state.step, total_steps, beta_start, beta_end)

        def loss_fn(params):
            return objective(params, epis_rng, policy_beta, hard_floor)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        train_state = train_state.apply_gradients(grads=grads)
        grad_mean = jnp.mean(jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.mean, grads))))
        grad_max = jnp.max(
            jnp.array(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads)))
        )
        return train_state, ((loss, (aux[0].welfare_loss, aux[1], aux[2])), (grad_mean, grad_max))

    def step_train(train_state, step_rng):
        epis_rng = random.split(step_rng, config["epis_per_step"])
        train_state, batch_metrics = jax.vmap(
            episode_train, in_axes=(None, 0), out_axes=(None, 0), axis_name="batch"
        )(train_state, jnp.stack(epis_rng))
        return train_state, batch_metrics

    def epoch_train(train_state, epoch_rng):
        epoch_rng, *step_rngs = random.split(epoch_rng, config["steps_per_epoch"] + 1)
        train_state, epoch_metrics = lax.scan(step_train, train_state, jnp.stack(step_rngs))
        return train_state, epoch_rng, epoch_metrics

    return epoch_train


def switching_boundary(env, apply_policy, params, grid_states, policy_beta, hard_floor=True, use_terminal_value=False):
    def latent_at(state):
        action = _policy_action(apply_policy, params, state, use_terminal_value)
        return env.econ._policy_latent(action)

    return jax.vmap(latent_at)(grid_states)


def create_exact_kink_eval_fn(env, apply_fn, config):
    objective, _ = create_exact_kink_objectives(env, apply_fn, config)
    total_steps = max(1, int(config["n_epochs"]) * int(config["steps_per_epoch"]))
    beta_start = float(config.get("policy_beta_start", 10.0))
    beta_end = float(config.get("policy_beta_end", 500.0))

    def eval_fn(train_state, rng):
        policy_beta = log_linear_beta(train_state.step, total_steps, beta_start, beta_end)
        loss, (metrics, value_loss, value_loss_perc) = objective(
            train_state.params, rng, policy_beta, False
        )
        return loss, metrics.welfare_loss, value_loss, value_loss_perc

    return eval_fn


def residual_by_slack_bin(residuals, slacks, occupancy=None):
    edges = jnp.array([-jnp.inf, -1e-12, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, jnp.inf])
    labels = (
        "violate",
        "0-0.001",
        "0.001-0.005",
        "0.005-0.01",
        "0.01-0.02",
        "0.02-0.05",
        "0.05-0.10",
        ">0.10",
    )
    weights = jnp.ones_like(residuals) if occupancy is None else occupancy
    weight_sum = jnp.sum(weights)
    rows = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        mask = (slacks > lo) & (slacks <= hi)
        mass = jnp.sum(weights * mask)
        safe = jnp.maximum(mass, 1e-12)
        mse = jnp.sum(weights * mask * residuals**2) / safe
        rows.append(
            {
                "bin": label,
                "mass": mass / jnp.maximum(weight_sum, 1e-12),
                "mse": mse,
                "rmse": jnp.sqrt(mse),
                "mse_contribution": jnp.sum(weights * mask * residuals**2) / jnp.maximum(weight_sum, 1e-12),
            }
        )
    return rows


def option_value_on_states(env, apply_policy, params, states, policy_beta, hard_floor=True, use_terminal_value=False):
    econ = env.econ

    def one_state(state):
        action = _policy_action(apply_policy, params, state, use_terminal_value)
        K, a = econ._capital_and_productivity(state)
        output = econ.production(K, a)
        investment, _, _ = econ.allocation_from_policy(action, output, None, policy_beta, hard_floor)

        def mu_node(node):
            a_next = econ.rho * a + econ.shock_sd * node
            state_next = econ._normalize_state(econ.next_capital(K, investment), a_next)
            action_next = _policy_action(apply_policy, params, state_next, use_terminal_value)
            return econ.multiplier_from_latent(
                econ._policy_latent(action_next), policy_beta, hard_floor
            )

        expected_mu = jnp.tensordot(_GH_WEIGHTS.astype(econ.precision), jax.vmap(mu_node)(_GH_NODES.astype(econ.precision)), axes=(0, 0))
        return econ.option_value(state, action, expected_mu, policy_beta, hard_floor)

    return jax.vmap(one_state)(states)
