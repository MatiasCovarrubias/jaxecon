from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax import random


class PlainDeqnEulerAux(NamedTuple):
    """Frozen economic quantities used by the direct Euler-moment update."""

    policy: jax.Array
    next_states: jax.Array
    next_policies: jax.Array
    continuation_realizations: jax.Array
    expected_continuation: jax.Array
    capital_price: jax.Array
    L: jax.Array
    M: jax.Array
    e_z: jax.Array
    cap_binding: jax.Array


def plain_deqn_envelope_surrogate(
    params,
    apply_fn,
    econ_model,
    batch_states,
    shocks,
    *,
    depth,
    sample_weights=None,
    normalizer=None,
    continuation_override_fn=None,
    continuation_override_mask=None,
):
    """Apply a finite-depth envelope correction to the Euler moment."""
    from DEQN.diagnostics.theory import finite_depth_policy_residual

    if int(depth) != depth or depth < 1:
        raise ValueError("depth must be a positive integer")
    if int(econ_model.n_sectors) != 1 or float(econ_model.i_min_frac) != 0:
        raise ValueError(
            "finite-depth envelope correction requires the smooth "
            "one-sector RBC"
        )
    batch_states = jnp.asarray(batch_states)
    shocks = jnp.asarray(shocks)
    if batch_states.ndim != 3 or shocks.ndim != 3:
        raise ValueError(
            "envelope correction expects states (paths, periods, state_dim) "
            "and shocks (paths, periods, shock_dim)"
        )
    paths, periods = batch_states.shape[:2]
    flat_states = batch_states.reshape((-1, batch_states.shape[-1]))
    flat_shocks = shocks.reshape((-1, shocks.shape[-1]))
    _, aux = plain_deqn_euler_surrogate(
        params,
        apply_fn,
        econ_model,
        flat_states,
        flat_shocks,
        sample_weights=None,
        normalizer=1,
        continuation_override_fn=continuation_override_fn,
        continuation_override_mask=(
            None
            if continuation_override_mask is None
            else jnp.asarray(continuation_override_mask).reshape(-1)
        ),
    )

    zero_shock = jnp.zeros((econ_model.n_sectors,), dtype=econ_model.precision)

    def local_terms(state):
        action = apply_fn(params, state)
        state_jacobian = jax.jacrev(
            lambda current: econ_model.step(
                current, action, zero_shock
            )[0]
        )(state)
        action_jacobian = jax.jacrev(
            lambda current: econ_model.step(
                state, current, zero_shock
            )[0]
        )(action)
        policy_jacobian = jax.jacrev(
            lambda current: apply_fn(params, current)[0]
        )(state)
        B = action_jacobian[0]
        A_yy = state_jacobian[0] + B * policy_jacobian[0]
        return B, A_yy

    current_states = flat_states
    next_states = aux.next_states[:, 0, :]
    B_current, _ = jax.vmap(local_terms)(current_states)
    B_next, A_yy_next = jax.vmap(local_terms)(next_states)
    safe_B_next = jnp.where(
        jnp.abs(B_next) > jnp.asarray(1e-12, dtype=B_next.dtype),
        B_next,
        jnp.ones_like(B_next),
    )
    operator = A_yy_next * B_current / safe_B_next
    operator = jnp.where(
        jnp.abs(B_next) > jnp.asarray(1e-12, dtype=B_next.dtype),
        operator,
        jnp.zeros_like(operator),
    )
    operator = operator.reshape((paths, periods))
    e_z = aux.e_z.reshape((paths, periods, -1))
    q_z = jax.vmap(
        lambda residual, bridge: finite_depth_policy_residual(
            residual,
            bridge,
            econ_model.discount_rate,
            depth,
        )
    )(e_z, operator)
    q_z = jax.lax.stop_gradient(q_z)
    policy = apply_fn(params, flat_states).reshape(
        (paths, periods, -1)
    )
    moments = jnp.sum(q_z * policy, axis=-1)
    if sample_weights is None:
        sample_weights = jnp.ones_like(moments)
    else:
        sample_weights = jax.lax.stop_gradient(
            jnp.asarray(sample_weights, dtype=moments.dtype)
        )
        if sample_weights.shape != moments.shape:
            raise ValueError("sample_weights must have shape (paths, periods)")
    if normalizer is None:
        normalizer = paths
    loss = -jnp.sum(sample_weights * moments) / jnp.asarray(
        normalizer, dtype=moments.dtype
    )
    return loss, aux


def plain_deqn_euler_surrogate(
    params,
    apply_fn,
    econ_model,
    batch_states,
    shocks,
    *,
    sample_weights=None,
    normalizer=None,
    continuation_override_fn=None,
    continuation_override_mask=None,
):
    """Return the direct Euler-moment surrogate for a frozen state batch.

    ``shocks`` has shape ``(batch, shock_dim)`` for one realized next shock
    per state, matching an APG path, or ``(batch, draws, shock_dim)`` for an
    independent shock bank at each state. The loss is
    ``-mean_batch(sum(stop_gradient(e_z) * z_theta, axis=-1))``, where
    ``e_z = -M * L`` and ``L = p - beta * E[Lambda_next]``.

    ``sample_weights`` can supply discounted occupancy weights. By default the
    loss is averaged over states. Passing ``normalizer`` permits a discounted
    sum divided by the number of independent trajectories, which matches the
    scale of APG's mean finite-horizon return.

    Every field in the returned auxiliary tuple is stopped. Consequently,
    minimizing the surrogate differentiates only the current policy carrier
    and applies the ascent vector ``mean(psi.T @ e_z)``.
    """
    batch_states = jax.lax.stop_gradient(jnp.asarray(batch_states))
    shocks = jax.lax.stop_gradient(jnp.asarray(shocks))
    if batch_states.ndim != 2:
        raise ValueError("batch_states must have shape (batch, state_dim)")
    if shocks.ndim == 2:
        shock_bank = shocks[:, None, :]
    elif shocks.ndim == 3:
        shock_bank = shocks
    else:
        raise ValueError(
            "shocks must have shape (batch, shock_dim) or "
            "(batch, draws, shock_dim)"
        )
    if shock_bank.shape[0] != batch_states.shape[0]:
        raise ValueError("shocks and batch_states must have the same batch size")
    if shock_bank.shape[1] < 1:
        raise ValueError("the shock bank must contain at least one draw per state")

    policy = apply_fn(params, batch_states)

    def next_states_for_state(state, action, state_shocks):
        return jax.vmap(lambda shock: econ_model.step(state, action, shock))(
            state_shocks
        )

    next_states = jax.vmap(next_states_for_state)(
        batch_states, policy, shock_bank
    )
    flat_next_states = next_states.reshape((-1, next_states.shape[-1]))
    flat_next_policies = apply_fn(params, flat_next_states)
    next_policies = flat_next_policies.reshape(
        next_states.shape[:-1] + (flat_next_policies.shape[-1],)
    )
    continuation_realizations = jax.vmap(
        jax.vmap(econ_model.expect_realization)
    )(next_states, next_policies)
    expected_continuation = jnp.mean(continuation_realizations, axis=1)
    if continuation_override_fn is not None:
        if continuation_override_mask is None:
            raise ValueError(
                "continuation_override_mask is required with an override"
            )
        override_mask = jnp.asarray(
            continuation_override_mask, dtype=bool
        )
        if override_mask.shape != (batch_states.shape[0],):
            raise ValueError(
                "continuation_override_mask must have shape (batch,)"
            )
        flat_override = continuation_override_fn(
            next_states.reshape((-1, next_states.shape[-1]))
        )
        override_values = flat_override.reshape(
            next_states.shape[:2] + (flat_override.shape[-1],)
        ).mean(axis=1)
        expected_continuation = jnp.where(
            override_mask[:, None],
            override_values,
            expected_continuation,
        )
    capital_price = econ_model.capital_price(batch_states, policy)
    L = capital_price - econ_model.beta * expected_continuation
    mapping = econ_model.action_euler_mapping(batch_states, policy)
    M = mapping.M
    e_z = -M * L

    aux = PlainDeqnEulerAux(
        policy=policy,
        next_states=next_states,
        next_policies=next_policies,
        continuation_realizations=continuation_realizations,
        expected_continuation=expected_continuation,
        capital_price=capital_price,
        L=L,
        M=M,
        e_z=e_z,
        cap_binding=mapping.cap_binding,
    )
    aux = jax.tree_util.tree_map(jax.lax.stop_gradient, aux)
    moments = jnp.sum(aux.e_z * policy, axis=-1)
    if sample_weights is None:
        sample_weights = jnp.ones_like(moments)
    else:
        sample_weights = jax.lax.stop_gradient(
            jnp.asarray(sample_weights, dtype=moments.dtype)
        )
        if sample_weights.shape != moments.shape:
            raise ValueError("sample_weights must have shape (batch,)")
    if normalizer is None:
        normalizer = moments.size
    normalizer = jnp.asarray(normalizer, dtype=moments.dtype)
    loss = -jnp.sum(sample_weights * moments) / normalizer
    return loss, aux


def create_batch_loss_fn(econ_model, config):

    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
        """Loss function of a batch of obs."""
        period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
        batch_policies = train_state.apply_fn(params, batch_obs)

        def period_loss(obs, policy, period_mc_rng):
            """Loss function for an individual period."""
            mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
            mc_nextobs = jax.vmap(econ_model.step, in_axes=(None, None, 0))(obs, policy, mc_shocks)
            mc_nextpols = train_state.apply_fn(params, mc_nextobs)
            expect = jax.lax.stop_gradient(
                jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
            )
            mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy)
            return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

        # parallelize callculation of period_loss for the entire batch
        losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
        mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        mean_accuracy = jnp.mean(mean_accuracies)
        min_accuracy = jnp.min(min_accuracies)
        mean_accs_foc = jnp.mean(mean_accs_foc, axis=0)  # average across period for each set of focs
        min_accs_foc = jnp.min(min_accs_foc, axis=0)
        loss_metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc
        return mean_loss, loss_metrics

    return batch_loss_fn


def create_batch_loss_fn_nostopgrad(econ_model, config):

    def batch_loss_fn(params, train_state, batch_obs, loss_rng):
        """Loss function of a batch of obs."""
        period_mc_rngs = random.split(loss_rng, batch_obs.shape[0])
        batch_policies = train_state.apply_fn(params, batch_obs)

        def period_loss(obs, policy, period_mc_rng):
            """Loss function for an individual period."""
            mc_shocks = econ_model.mc_shocks(period_mc_rng, config["mc_draws"])
            mc_nextobs = jax.vmap(econ_model.step, in_axes=(None, None, 0))(obs, policy, mc_shocks)
            mc_nextpols = train_state.apply_fn(params, mc_nextobs)
            expect = jnp.mean(jax.vmap(econ_model.expect_realization)(mc_nextobs, mc_nextpols), axis=0)
            mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc = econ_model.loss(obs, expect, policy)
            return mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc

        # parallelize callculation of period_loss for the entire batch
        losses_metrics = jax.vmap(period_loss)(batch_obs, batch_policies, jnp.stack(period_mc_rngs))
        mean_losses, mean_accuracies, min_accuracies, mean_accs_foc, min_accs_foc = losses_metrics
        mean_loss = jnp.mean(mean_losses)  # average accross periods
        mean_accuracy = jnp.mean(mean_accuracies)
        min_accuracy = jnp.min(min_accuracies)
        mean_accs_foc = jnp.mean(mean_accs_foc, axis=0)  # average across period for each set of focs
        min_accs_foc = jnp.min(min_accs_foc, axis=0)
        loss_metrics = mean_loss, mean_accuracy, min_accuracy, mean_accs_foc, min_accs_foc
        return mean_loss, loss_metrics

    return batch_loss_fn
