"""Primal-dual APG for the one-sector irreversible RBC model."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze
from jax import lax, random

from DEQN.algorithm.simulation import sample_episode_shocks


class ConstrainedObjectiveMetrics(NamedTuple):
    actor_loss: jax.Array
    true_return: jax.Array
    augmented_return: jax.Array
    dual_loss: jax.Array
    occupancy_violation_frac: jax.Array
    occupancy_shortfall_mean: jax.Array
    occupancy_shortfall_max: jax.Array
    grid_violation_frac: jax.Array
    grid_shortfall_mean: jax.Array
    grid_shortfall_max: jax.Array
    multiplier_mean: jax.Array
    multiplier_max: jax.Array
    complementarity_mean: jax.Array


class ConstrainedTrainMetrics(NamedTuple):
    objective: ConstrainedObjectiveMetrics
    actor_grad_norm: jax.Array
    multiplier_grad_norm: jax.Array


class ConstrainedTrainState(NamedTuple):
    actor: object
    multiplier: object


def _stop_tree(tree):
    return jax.tree_util.tree_map(lax.stop_gradient, tree)


def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def create_constrained_objectives(
    env,
    actor_apply_fn,
    multiplier_apply_fn,
    config,
):
    """Create detached actor-max and multiplier-min objectives."""
    if env.N != 1:
        raise ValueError("learned-multiplier APG currently supports one sector")
    if env.econ.project_investment:
        raise ValueError("learned-multiplier APG requires unprojected investment")
    if env.investment_penalty:
        raise ValueError("learned-multiplier APG cannot use a fixed investment penalty")

    horizon = int(config["periods_per_epis"])
    init_range = config.get("init_range", 5)
    simul_vol_scale = config.get("simul_vol_scale", 1.0)
    antithetic = bool(config.get("antithetic_episodes", False))
    rho = float(config.get("augmented_lagrangian_rho", 1.0))
    grid_share = float(config.get("constraint_grid_share", 0.5))
    violation_weight = float(config.get("multiplier_violation_weight", 1.0))
    multiplier_update = config.get("multiplier_update_mode", "lagrangian")
    augmented_form = config.get(
        "augmented_lagrangian_form",
        "hinge_quadratic",
    )
    projected_step_size = float(config.get("multiplier_projected_step_size", 0.01))
    multiplier_min_value = float(config.get("multiplier_min_value", 1e-8))
    if not 0 <= grid_share <= 1:
        raise ValueError("constraint_grid_share must be between zero and one")
    if rho < 0:
        raise ValueError("augmented_lagrangian_rho must be nonnegative")
    if augmented_form not in ("hinge_quadratic", "phr"):
        raise ValueError(
            "augmented_lagrangian_form must be 'hinge_quadratic' or 'phr'"
        )
    if augmented_form == "phr" and rho <= 0:
        raise ValueError("PHR augmented Lagrangian requires positive rho")
    if violation_weight < 1:
        raise ValueError("multiplier_violation_weight must be at least one")
    if multiplier_update not in ("lagrangian", "projected", "phr_direct"):
        raise ValueError(
            "multiplier_update_mode must be 'lagrangian', 'projected', or "
            "'phr_direct'"
        )
    if multiplier_update == "phr_direct":
        if augmented_form != "phr":
            raise ValueError("phr_direct multiplier updates require the PHR actor form")
        if config.get("multiplier_architecture") != "grid":
            raise ValueError("phr_direct multiplier updates require a grid multiplier")
        if violation_weight != 1:
            raise ValueError(
                "phr_direct multiplier updates use canonical PHR scaling and "
                "require multiplier_violation_weight=1"
            )
        if int(config.get("multiplier_warmup_steps", 0)) != 0:
            raise ValueError("phr_direct multiplier updates do not support warmup")
    if projected_step_size <= 0:
        raise ValueError("multiplier_projected_step_size must be positive")
    if multiplier_min_value <= 0:
        raise ValueError("multiplier_min_value must be positive")
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
    if use_model_terminal_value and config.get("use_terminal_value"):
        raise ValueError(
            "learned critic bootstrap and model terminal continuation cannot be combined"
        )
    if terminal_value_horizon < 1:
        raise ValueError("terminal_value_horizon must be positive")

    grid_states = env.constraint_state_grid(
        size=int(config.get("constraint_grid_size", 33)),
        k_min=float(config.get("constraint_grid_k_min", 0.90)),
        k_max=float(config.get("constraint_grid_k_max", 1.16)),
        a_sd_min=float(config.get("constraint_grid_a_sd_min", -2.5)),
        a_sd_max=float(config.get("constraint_grid_a_sd_max", 2.5)),
    )
    utility_scale = jnp.reshape(env.constraint_utility_scale(), ())
    discounted_weight = jnp.sum(
        jnp.power(env.discount_rate, jnp.arange(horizon, dtype=env.econ.precision))
    )

    def path_statistics(actor_params, multiplier_params, init_obs, shocks):
        initial = (
            init_obs,
            jnp.zeros((), dtype=env.econ.precision),
            jnp.zeros((), dtype=env.econ.precision),
            jnp.zeros((), dtype=env.econ.precision),
            jnp.ones((), dtype=env.econ.precision),
        )

        def period_step(carry, shock):
            obs, true_return, augmented_constraint, dual_constraint, discount = carry
            action = actor_apply_fn(actor_params, obs)
            multiplier = jnp.reshape(multiplier_apply_fn(multiplier_params, obs), ())
            slack = jnp.reshape(env.normalized_investment_slack(obs, action), ())
            shortfall = jnp.maximum(-slack, 0)
            reward = jnp.reshape(env.econ.reward(obs, action), ())
            if augmented_form == "phr":
                projected_multiplier = jnp.maximum(multiplier - rho * slack, 0)
                actor_constraint = -0.5 / rho * (
                    jnp.square(projected_multiplier) - jnp.square(multiplier)
                )
            else:
                actor_constraint = (
                    multiplier * slack - 0.5 * rho * jnp.square(shortfall)
                )
            dual_weight = jnp.where(slack < 0, violation_weight, 1.0)
            if multiplier_update in ("projected", "phr_direct"):
                dual_step = (
                    rho
                    if multiplier_update == "phr_direct"
                    else projected_step_size
                )
                dual_weight = (
                    jnp.ones_like(dual_weight)
                    if multiplier_update == "phr_direct"
                    else dual_weight
                )
                multiplier_logits = jnp.reshape(
                    multiplier_apply_fn(
                        multiplier_params,
                        obs,
                        return_logits=True,
                    ),
                    (),
                )
                target_multiplier = jnp.maximum(
                    lax.stop_gradient(
                        multiplier - dual_step * dual_weight * slack
                    ),
                    multiplier_min_value,
                )
                target_logits = jnp.where(
                    target_multiplier > 20,
                    target_multiplier,
                    jnp.log(jnp.expm1(target_multiplier)),
                )
                dual_term = 0.5 * jnp.square(
                    multiplier_logits - lax.stop_gradient(target_logits)
                )
            else:
                dual_term = multiplier * lax.stop_gradient(dual_weight * slack)
            obs_next = env.econ.step(obs, action, shock)
            metrics = jnp.stack(
                [
                    shortfall > 0,
                    shortfall,
                    multiplier,
                    multiplier * jnp.maximum(slack, 0),
                ]
            )
            next_carry = (
                obs_next,
                true_return + discount * reward,
                augmented_constraint + discount * actor_constraint,
                dual_constraint + discount * dual_term,
                discount * env.discount_rate,
            )
            return next_carry, metrics

        final, metrics = lax.scan(period_step, initial, shocks)
        true_return = final[1]
        if use_model_terminal_value:
            true_return = true_return + final[4] * env.terminal_value(
                final[0],
                terminal_value_horizon,
            )
        return true_return, final[2], final[3], metrics

    def episode_statistics(actor_params, multiplier_params, key):
        init_obs = env.econ.initial_state(key, init_range)
        shocks = sample_episode_shocks(env.econ, key, horizon, simul_vol_scale)
        plus = path_statistics(actor_params, multiplier_params, init_obs, shocks)
        if not antithetic:
            return plus
        minus = path_statistics(actor_params, multiplier_params, init_obs, -shocks)
        return jax.tree_util.tree_map(lambda x, y: 0.5 * (x + y), plus, minus)

    def aggregate(actor_params, multiplier_params, episode_keys):
        true_returns, occupancy_augmented, occupancy_dual, occupancy_metrics = jax.vmap(
            episode_statistics,
            in_axes=(None, None, 0),
        )(actor_params, multiplier_params, episode_keys)

        grid_actions = actor_apply_fn(actor_params, grid_states)
        grid_slack = env.normalized_investment_slack(grid_states, grid_actions).reshape(-1)
        grid_multiplier = multiplier_apply_fn(multiplier_params, grid_states).reshape(-1)
        grid_shortfall = jnp.maximum(-grid_slack, 0)
        if augmented_form == "phr":
            grid_projected_multiplier = jnp.maximum(
                grid_multiplier - rho * grid_slack,
                0,
            )
            grid_augmented = jnp.mean(
                -0.5
                / rho
                * (
                    jnp.square(grid_projected_multiplier)
                    - jnp.square(grid_multiplier)
                )
            )
        else:
            grid_augmented = jnp.mean(
                grid_multiplier * grid_slack
                - 0.5 * rho * jnp.square(grid_shortfall)
            )
        grid_dual_weight = jnp.where(grid_slack < 0, violation_weight, 1.0)
        if multiplier_update in ("projected", "phr_direct"):
            grid_dual_step = (
                rho if multiplier_update == "phr_direct" else projected_step_size
            )
            grid_dual_weight = (
                jnp.ones_like(grid_dual_weight)
                if multiplier_update == "phr_direct"
                else grid_dual_weight
            )
            grid_multiplier_logits = multiplier_apply_fn(
                multiplier_params,
                grid_states,
                return_logits=True,
            ).reshape(-1)
            grid_target_multiplier = jnp.maximum(
                lax.stop_gradient(
                    grid_multiplier
                    - grid_dual_step * grid_dual_weight * grid_slack
                ),
                multiplier_min_value,
            )
            grid_target_logits = jnp.where(
                grid_target_multiplier > 20,
                grid_target_multiplier,
                jnp.log(jnp.expm1(grid_target_multiplier)),
            )
            grid_dual = jnp.mean(
                0.5
                * jnp.square(
                    grid_multiplier_logits - lax.stop_gradient(grid_target_logits)
                )
            )
        else:
            grid_dual = jnp.mean(
                grid_multiplier * lax.stop_gradient(grid_dual_weight * grid_slack)
            )

        occupancy_augmented_mean = jnp.mean(occupancy_augmented)
        occupancy_dual_mean = jnp.mean(occupancy_dual)
        mixed_augmented = (
            (1 - grid_share) * occupancy_augmented_mean
            + grid_share * discounted_weight * grid_augmented
        )
        mixed_dual = (
            (1 - grid_share) * occupancy_dual_mean
            + grid_share * discounted_weight * grid_dual
        )
        true_return = jnp.mean(true_returns)
        augmented_return = true_return + utility_scale * mixed_augmented

        occupancy_metrics = occupancy_metrics.reshape(-1, 4)
        occupancy_shortfall = occupancy_metrics[:, 1]
        occupancy_multiplier = occupancy_metrics[:, 2]
        multiplier_mean = (
            (1 - grid_share) * jnp.mean(occupancy_multiplier)
            + grid_share * jnp.mean(grid_multiplier)
        )
        multiplier_max = jnp.maximum(
            jnp.max(occupancy_multiplier),
            jnp.max(grid_multiplier),
        )
        complementarity_mean = (
            (1 - grid_share) * jnp.mean(occupancy_metrics[:, 3])
            + grid_share * jnp.mean(grid_multiplier * jnp.maximum(grid_slack, 0))
        )
        metrics = ConstrainedObjectiveMetrics(
            actor_loss=-augmented_return,
            true_return=true_return,
            augmented_return=augmented_return,
            dual_loss=utility_scale * mixed_dual,
            occupancy_violation_frac=jnp.mean(occupancy_metrics[:, 0]),
            occupancy_shortfall_mean=jnp.mean(occupancy_shortfall),
            occupancy_shortfall_max=jnp.max(occupancy_shortfall),
            grid_violation_frac=jnp.mean(grid_shortfall > 0),
            grid_shortfall_mean=jnp.mean(grid_shortfall),
            grid_shortfall_max=jnp.max(grid_shortfall),
            multiplier_mean=multiplier_mean,
            multiplier_max=multiplier_max,
            complementarity_mean=complementarity_mean,
        )
        return metrics

    def actor_objective(actor_params, multiplier_params, episode_keys):
        metrics = aggregate(
            actor_params,
            _stop_tree(multiplier_params),
            episode_keys,
        )
        return metrics.actor_loss, metrics

    def multiplier_objective(multiplier_params, actor_params, episode_keys):
        metrics = aggregate(
            _stop_tree(actor_params),
            multiplier_params,
            episode_keys,
        )
        return metrics.dual_loss, metrics

    return actor_objective, multiplier_objective


def create_constrained_epoch_train_fn(
    env,
    actor_apply_fn,
    multiplier_apply_fn,
    config,
):
    actor_objective, multiplier_objective = create_constrained_objectives(
        env,
        actor_apply_fn,
        multiplier_apply_fn,
        config,
    )
    episode_count = int(config["epis_per_step"])
    multiplier_update = config.get("multiplier_update_mode", "lagrangian")
    actor_steps = int(config.get("actor_steps_per_multiplier_update", 1))
    alternating = bool(config.get("alternate_actor_multiplier_updates", False))
    direct_grid_update = multiplier_update == "phr_direct"
    predictor_corrector = bool(
        config.get("multiplier_predictor_corrector", False)
    )
    if actor_steps < 1:
        raise ValueError("actor_steps_per_multiplier_update must be positive")
    if actor_steps != 1 and not (alternating or direct_grid_update):
        raise ValueError(
            "multiple actor steps require alternate_actor_multiplier_updates=True"
        )
    if predictor_corrector and not direct_grid_update:
        raise ValueError(
            "multiplier_predictor_corrector requires phr_direct updates"
        )

    if direct_grid_update:
        grid_size = int(config.get("constraint_grid_size", 33))
        stationary_a_sd = env.econ.shock_sd[0] / jnp.sqrt(
            1 - env.econ.rho[0] ** 2
        )
        capital_axis = jnp.linspace(
            jnp.log(float(config.get("constraint_grid_k_min", 0.90))),
            jnp.log(float(config.get("constraint_grid_k_max", 1.16))),
            grid_size,
            dtype=env.econ.precision,
        )
        productivity_axis = stationary_a_sd * jnp.linspace(
            float(config.get("constraint_grid_a_sd_min", -2.5)),
            float(config.get("constraint_grid_a_sd_max", 2.5)),
            grid_size,
            dtype=env.econ.precision,
        )
        capital_grid, productivity_grid = jnp.meshgrid(
            capital_axis,
            productivity_axis,
            indexing="ij",
        )
        grid_states = jnp.stack(
            [capital_grid.reshape(-1), productivity_grid.reshape(-1)],
            axis=-1,
        )
        rho = float(config["augmented_lagrangian_rho"])
        multiplier_min_value = float(config.get("multiplier_min_value", 1e-8))
        direct_relaxation_initial = float(
            config.get("multiplier_direct_relaxation", 1.0)
        )
        direct_relaxation_floor = float(
            config.get(
                "multiplier_direct_relaxation_floor",
                direct_relaxation_initial,
            )
        )
        direct_relaxation_decay = float(
            config.get("multiplier_direct_relaxation_decay", 1.0)
        )
        if not 0 < direct_relaxation_initial <= 1:
            raise ValueError("multiplier_direct_relaxation must be in (0, 1]")
        if not 0 < direct_relaxation_floor <= direct_relaxation_initial:
            raise ValueError(
                "multiplier_direct_relaxation_floor must be in "
                "(0, multiplier_direct_relaxation]"
            )
        if not 0 < direct_relaxation_decay <= 1:
            raise ValueError(
                "multiplier_direct_relaxation_decay must be in (0, 1]"
            )

        def direct_multiplier_parameters(multiplier_state, actor_params):
            grid_actions = actor_apply_fn(actor_params, grid_states)
            grid_slack = env.normalized_investment_slack(
                grid_states,
                grid_actions,
            ).reshape(grid_size, grid_size)
            current_multiplier = multiplier_apply_fn(
                multiplier_state.params,
                grid_states,
            ).reshape(grid_size, grid_size)
            canonical_target = jnp.maximum(
                current_multiplier - rho * grid_slack,
                multiplier_min_value,
            )
            direct_relaxation = jnp.maximum(
                direct_relaxation_floor,
                direct_relaxation_initial
                * jnp.power(
                    direct_relaxation_decay,
                    multiplier_state.step,
                ),
            )
            target_multiplier = (
                (1 - direct_relaxation) * current_multiplier
                + direct_relaxation * canonical_target
            )
            target_logits = jnp.where(
                target_multiplier > 20,
                target_multiplier,
                jnp.log(jnp.expm1(target_multiplier)),
            )
            mutable_params = unfreeze(multiplier_state.params)
            mutable_params["params"]["logits"] = target_logits
            next_params = (
                freeze(mutable_params)
                if isinstance(multiplier_state.params, FrozenDict)
                else mutable_params
            )
            update_norm = jnp.linalg.norm(target_multiplier - current_multiplier)
            return next_params, update_norm

        def direct_multiplier_update(multiplier_state, actor_params):
            next_params, update_norm = direct_multiplier_parameters(
                multiplier_state,
                actor_params,
            )
            return multiplier_state.replace(
                step=multiplier_state.step + 1,
                params=next_params,
            ), update_norm

    def actor_step(actor_state, inputs):
        multiplier_params, actor_rng = inputs
        episode_keys = random.split(actor_rng, episode_count)
        (_, metrics), actor_grads = jax.value_and_grad(
            actor_objective,
            argnums=0,
            has_aux=True,
        )(actor_state.params, multiplier_params, episode_keys)
        return actor_state.apply_gradients(grads=actor_grads), (
            metrics,
            _tree_l2_norm(actor_grads),
        )

    def step_train(state, step_rng):
        if alternating or direct_grid_update:
            step_rngs = random.split(step_rng, actor_steps + 1)
            actor_multiplier_params = state.multiplier.params
            if predictor_corrector:
                actor_multiplier_params, _ = direct_multiplier_parameters(
                    state.multiplier,
                    state.actor.params,
                )
            multiplier_params = jax.tree_util.tree_map(
                lambda value: jnp.broadcast_to(
                    value,
                    (actor_steps,) + value.shape,
                ),
                actor_multiplier_params,
            )
            next_actor, (actor_metrics, actor_grad_norms) = lax.scan(
                actor_step,
                state.actor,
                (multiplier_params, step_rngs[:-1]),
            )
            actor_metrics = jax.tree_util.tree_map(jnp.mean, actor_metrics)
            episode_keys = random.split(step_rngs[-1], episode_count)
            if direct_grid_update:
                (_, multiplier_metrics) = multiplier_objective(
                    state.multiplier.params,
                    next_actor.params,
                    episode_keys,
                )
                next_multiplier, multiplier_update_norm = direct_multiplier_update(
                    state.multiplier,
                    next_actor.params,
                )
            else:
                (_, multiplier_metrics), multiplier_grads = jax.value_and_grad(
                    multiplier_objective,
                    argnums=0,
                    has_aux=True,
                )(state.multiplier.params, next_actor.params, episode_keys)
                next_multiplier = state.multiplier.apply_gradients(
                    grads=multiplier_grads
                )
                multiplier_update_norm = _tree_l2_norm(multiplier_grads)
            next_state = ConstrainedTrainState(
                actor=next_actor,
                multiplier=next_multiplier,
            )
            metrics = ConstrainedTrainMetrics(
                objective=actor_metrics._replace(
                    dual_loss=multiplier_metrics.dual_loss
                ),
                actor_grad_norm=jnp.mean(actor_grad_norms),
                multiplier_grad_norm=multiplier_update_norm,
            )
            return next_state, metrics

        episode_keys = random.split(step_rng, episode_count)
        (_, actor_metrics), actor_grads = jax.value_and_grad(
            actor_objective,
            argnums=0,
            has_aux=True,
        )(state.actor.params, state.multiplier.params, episode_keys)
        (_, multiplier_metrics), multiplier_grads = jax.value_and_grad(
            multiplier_objective,
            argnums=0,
            has_aux=True,
        )(state.multiplier.params, state.actor.params, episode_keys)
        next_state = ConstrainedTrainState(
            actor=state.actor.apply_gradients(grads=actor_grads),
            multiplier=state.multiplier.apply_gradients(grads=multiplier_grads),
        )
        metrics = ConstrainedTrainMetrics(
            objective=actor_metrics._replace(dual_loss=multiplier_metrics.dual_loss),
            actor_grad_norm=_tree_l2_norm(actor_grads),
            multiplier_grad_norm=_tree_l2_norm(multiplier_grads),
        )
        return next_state, metrics

    def epoch_train(state, epoch_rng):
        epoch_rng, *step_rngs = random.split(
            epoch_rng,
            int(config["steps_per_epoch"]) + 1,
        )
        state, metrics = lax.scan(step_train, state, jnp.stack(step_rngs))
        return state, epoch_rng, metrics

    return epoch_train


def create_constrained_eval_fn(
    env,
    actor_apply_fn,
    multiplier_apply_fn,
    config,
):
    eval_config = {
        **config,
        "periods_per_epis": config.get(
            "eval_periods_per_epis",
            config["periods_per_epis"],
        ),
        "antithetic_episodes": False,
    }
    actor_objective, multiplier_objective = create_constrained_objectives(
        env,
        actor_apply_fn,
        multiplier_apply_fn,
        eval_config,
    )
    episode_count = int(config.get("eval_n_epis", 1024))

    def evaluate(state, eval_rng):
        episode_keys = random.split(eval_rng, episode_count)
        (_, actor_metrics), actor_grads = jax.value_and_grad(
            actor_objective,
            argnums=0,
            has_aux=True,
        )(state.actor.params, state.multiplier.params, episode_keys)
        (_, multiplier_metrics), multiplier_grads = jax.value_and_grad(
            multiplier_objective,
            argnums=0,
            has_aux=True,
        )(state.multiplier.params, state.actor.params, episode_keys)
        return ConstrainedTrainMetrics(
            objective=actor_metrics._replace(dual_loss=multiplier_metrics.dual_loss),
            actor_grad_norm=_tree_l2_norm(actor_grads),
            multiplier_grad_norm=_tree_l2_norm(multiplier_grads),
        )

    return evaluate


def create_multiplier_warmup_fn(
    env,
    actor_apply_fn,
    multiplier_apply_fn,
    config,
):
    """Fit the state-dependent dual response before actor updates begin."""
    _, multiplier_objective = create_constrained_objectives(
        env,
        actor_apply_fn,
        multiplier_apply_fn,
        config,
    )
    episode_count = int(config["epis_per_step"])
    warmup_steps = int(config.get("multiplier_warmup_steps", 0))

    def warmup_step(state, step_rng):
        episode_keys = random.split(step_rng, episode_count)
        (_, metrics), grads = jax.value_and_grad(
            multiplier_objective,
            argnums=0,
            has_aux=True,
        )(state.multiplier.params, state.actor.params, episode_keys)
        next_state = ConstrainedTrainState(
            actor=state.actor,
            multiplier=state.multiplier.apply_gradients(grads=grads),
        )
        return next_state, metrics

    def warmup(state, warmup_rng):
        if warmup_steps == 0:
            return state, warmup_rng, None
        warmup_rng, *step_rngs = random.split(warmup_rng, warmup_steps + 1)
        state, metrics = lax.scan(warmup_step, state, jnp.stack(step_rngs))
        return state, warmup_rng, metrics

    return warmup
