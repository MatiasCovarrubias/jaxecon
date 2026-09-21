"""Small policy used by the neural trainers."""

import jax
from jax import numpy as jnp
from jax import random


def init_policy(rng, model, hidden):
    """Affine-tanh network. A zero output is the steady-state control."""
    state_dim = model.state_ss.shape[0]
    control_dim = model.control_ss.shape[0]
    k1, k2 = random.split(rng)
    scale = jnp.asarray(0.1, dtype=model.state_ss.dtype)
    return {
        "w1": scale * random.normal(k1, (state_dim, hidden), dtype=model.state_ss.dtype),
        "b1": jnp.zeros((hidden,), dtype=model.state_ss.dtype),
        "w2": scale * random.normal(k2, (hidden, control_dim), dtype=model.state_ss.dtype),
        "b2": jnp.zeros((control_dim,), dtype=model.state_ss.dtype),
    }


def policy_control(params, state, model, width):
    """Map a state to a control in a band around the steady-state control."""
    norm = (state - model.state_ss) / model.state_sd
    hidden = jnp.tanh(norm @ params["w1"] + params["b1"])
    raw = hidden @ params["w2"] + params["b2"]
    width = jnp.asarray(width, dtype=raw.dtype)
    return model.control_ss + width * jnp.tanh(raw)


def sgd(step, params, rng, n_steps, learning_rate):
    """Compiled gradient step, scanned from Python once per update."""
    rate = jnp.asarray(learning_rate)

    def update(current, key):
        loss, grads = step(current, key)
        leaves = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
        updated = jax.tree_util.tree_map(lambda value, grad: value - rate * grad, current, grads)
        return updated, loss, grad_norm

    update = jax.jit(update)
    loss = grad_norm = None
    for _ in range(n_steps):
        rng, key = random.split(rng)
        params, loss, grad_norm = update(params, key)
    return params, {"loss": float(loss), "grad_norm": float(grad_norm)}
