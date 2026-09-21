"""ReLU policy and Adam step shared by APG and DEQN.

The network reads the normalized state ``(state - state_ss) / state_sd`` and
returns a logit. A zero logit is the steady-state saving rate, mapped through
a sigmoid into ``(1e-6, 1 - 1e-6)``.
"""

import time

import optax
import jax
from jax import numpy as jnp
from jax import random

SAVING_MIN = 1e-6
SAVING_MAX = 1.0 - 1e-6


def split_streams(seed):
    """Split one seed into ``train``, ``init``, and ``eval``.

    Training shocks are drawn from ``train`` and scoring from ``eval``.
    Neither key depends on the other, so APG and DEQN see the same shocks.
    """
    train, init, eval_rng = random.split(random.PRNGKey(int(seed)), 3)
    return {"train": train, "init": init, "eval": eval_rng}


def batch_draw(model, key, episodes, periods):
    """Initial states and shock paths for one update. Shared by APG and DEQN."""

    def one(episode_key):
        state_key, shock_key = random.split(episode_key)
        state = model.initial_state(state_key)
        shocks = jax.vmap(model.sample_shock)(random.split(shock_key, periods))
        return state, shocks

    return jax.vmap(one)(random.split(key, episodes))


def _widths(hidden):
    if isinstance(hidden, int):
        return (int(hidden),)
    return tuple(int(width) for width in hidden)


def init_policy(rng, model, hidden):
    """Two-hidden-layer ReLU net unless ``hidden`` says otherwise. Output bias is zero."""
    dtype = model.state_ss.dtype
    dims = (int(model.state_ss.shape[0]), *_widths(hidden), int(model.control_ss.shape[0]))
    layers = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        rng, key = random.split(rng)
        scale = jnp.sqrt(jnp.asarray(1.0 / fan_in, dtype=dtype))
        layers.append(
            {
                "w": scale * random.normal(key, (fan_in, fan_out), dtype=dtype),
                "b": jnp.zeros((fan_out,), dtype=dtype),
            }
        )
    return layers


def logit_offset(model):
    """Logit shift that makes a zero network output the steady-state saving rate."""
    saving = model.control_ss[0]
    scaled = (saving - SAVING_MIN) / (SAVING_MAX - SAVING_MIN)
    return jnp.log(scaled) - jnp.log1p(-scaled)


def policy_logit(params, state, model):
    """Normalized state to a scalar logit."""
    hidden = (state - model.state_ss) / model.state_sd
    for index, layer in enumerate(params):
        hidden = hidden @ layer["w"] + layer["b"]
        if index + 1 < len(params):
            hidden = jnp.maximum(hidden, 0.0)
    return hidden[..., 0]


def saving_rate(logit, model):
    """Map a logit to a saving rate. Zero is the steady-state rate."""
    dtype = logit.dtype
    lower = jnp.asarray(SAVING_MIN, dtype=dtype)
    upper = jnp.asarray(SAVING_MAX, dtype=dtype)
    unit = jax.nn.sigmoid(logit + logit_offset(model).astype(dtype))
    return lower + (upper - lower) * unit


def policy_control(params, state, model):
    """Saving rate at one state, shaped like ``control_ss``."""
    return saving_rate(policy_logit(params, state, model), model)[None]


def training_seconds(first, rest, steps):
    """Split a timed loop into compilation and the remaining run.

    ``first`` is the first call, which includes compilation. ``rest`` is every
    later call. The compile estimate is how much the first call exceeded one
    later step.
    """
    step = rest / (steps - 1) if steps > 1 else first
    compile_seconds = max(0.0, first - step) if steps > 1 else 0.0
    run_seconds = step * steps if steps > 1 else first
    return compile_seconds, run_seconds


def adam(step, params, rng, n_steps, learning_rate, cosine_alpha):
    """Adam with a cosine schedule that ends at ``cosine_alpha`` times the initial rate."""
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(int(n_steps), 1),
        alpha=cosine_alpha,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    def update(current, opt_state, key):
        loss, grads = step(current, key)
        updates, opt_state = optimizer.update(grads, opt_state, current)
        current = optax.apply_updates(current, updates)
        grad_sq = sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(grads))
        return current, opt_state, loss, jnp.sqrt(grad_sq)

    update = jax.jit(update)
    n_steps = int(n_steps)
    rng, key = random.split(rng)
    tick = time.perf_counter()
    params, opt_state, loss, grad_norm = update(params, opt_state, key)
    jax.block_until_ready(loss)
    first = time.perf_counter() - tick
    tick = time.perf_counter()
    for _ in range(n_steps - 1):
        rng, key = random.split(rng)
        params, opt_state, loss, grad_norm = update(params, opt_state, key)
    if n_steps > 1:
        jax.block_until_ready(loss)
    rest = time.perf_counter() - tick
    compile_seconds, run_seconds = training_seconds(first, rest, n_steps)
    return params, {
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
    }
