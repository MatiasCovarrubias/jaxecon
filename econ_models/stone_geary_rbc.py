"""One-sector Stone-Geary RBC in the shared model contract.

Coordinates are the model's own: state ``[log K, log TFP]``, control the
saving rate. ``state_sd`` is ones, so a caller normalizes by subtracting the
steady state. ``initial_state`` draws capital from ``[0.5, 1.5] K_ss`` and
productivity from ``[0.85, 1.15]``.
"""

from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)


class Params(NamedTuple):
    beta: jnp.ndarray
    alpha: jnp.ndarray
    delta: jnp.ndarray
    sigma: jnp.ndarray
    rho: jnp.ndarray
    shock_sd: jnp.ndarray
    cbar_frac: jnp.ndarray
    phi: jnp.ndarray


class SteadyState(NamedTuple):
    state: jnp.ndarray
    control: jnp.ndarray
    K: jnp.ndarray
    Y: jnp.ndarray
    C: jnp.ndarray
    I: jnp.ndarray
    c_bar: jnp.ndarray


class StoneGearyRbc:
    """Planner RBC with subsistence and a quadratic capital adjustment cost.

    ``expectation`` is the realized Euler continuation. ``residuals`` is
    ``q / (beta * E) - 1``, with ``q`` the marginal value of capital.
    """

    n_endogenous = 1

    def __init__(
        self,
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        sigma=2.0,
        rho=0.9,
        shock_sd=0.07,
        cbar_frac=0.35,
        phi=2.0,
        precision=jnp.float64,
    ):
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < delta <= 1.0:
            raise ValueError("delta must lie in (0, 1]")
        if sigma <= 0.0 or abs(sigma - 1.0) < 1e-6:
            raise ValueError("sigma must be positive and different from one")
        if not abs(rho) < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if shock_sd < 0.0:
            raise ValueError("shock_sd must be non-negative")
        if not 0.0 <= cbar_frac < 1.0:
            raise ValueError("cbar_frac must lie in [0, 1)")
        if phi < 0.0:
            raise ValueError("phi must be non-negative")

        dtype = precision
        self.params = Params(
            beta=jnp.asarray(beta, dtype=dtype),
            alpha=jnp.asarray(alpha, dtype=dtype),
            delta=jnp.asarray(delta, dtype=dtype),
            sigma=jnp.asarray(sigma, dtype=dtype),
            rho=jnp.asarray(rho, dtype=dtype),
            shock_sd=jnp.asarray(shock_sd, dtype=dtype),
            cbar_frac=jnp.asarray(cbar_frac, dtype=dtype),
            phi=jnp.asarray(phi, dtype=dtype),
        )
        self.discount_rate = self.params.beta
        self._steady = _steady_state(self.params)
        self.state_ss = self._steady.state
        self.control_ss = self._steady.control
        self.state_sd = jnp.ones_like(self.state_ss)
        self.control_sd = jnp.ones_like(self.control_ss)
        self.c_bar = self._steady.c_bar

    def steady_state(self):
        return self._steady

    def initial_state(self, rng):
        dtype = self.state_ss.dtype
        rng_k, rng_a = random.split(rng)
        capital = random.uniform(
            rng_k,
            (),
            minval=0.5 * self._steady.K,
            maxval=1.5 * self._steady.K,
            dtype=dtype,
        )
        productivity = random.uniform(
            rng_a,
            (),
            minval=jnp.asarray(0.85, dtype=dtype),
            maxval=jnp.asarray(1.15, dtype=dtype),
            dtype=dtype,
        )
        return jnp.stack([jnp.log(capital), jnp.log(productivity)])

    def sample_shock(self, rng):
        return random.normal(rng, (1,), dtype=self.state_ss.dtype)

    def transition(self, state, control, shock):
        capital, _, investment, _, _ = self._allocation(state, control)
        gap = investment / capital - self.params.delta
        adjustment = 0.5 * self.params.phi * gap**2 * capital
        capital_next = (1.0 - self.params.delta) * capital + investment - adjustment
        productivity = state[..., 1]
        productivity_next = self.params.rho * productivity + self.params.shock_sd * shock[..., 0]
        return jnp.stack([jnp.log(capital_next), productivity_next], axis=-1)

    def utility(self, state, control):
        *_, consumption, _ = self._allocation(state, control)
        excess = self._excess(consumption)
        sigma = self.params.sigma
        return excess ** (1.0 - sigma) / (1.0 - sigma)

    def expectation(self, state, control):
        capital, productivity, investment, consumption, _ = self._allocation(state, control)
        marginal = self._marginal_utility(consumption)
        marginal_product = self.params.alpha * jnp.exp(productivity) * capital ** (self.params.alpha - 1.0)
        price = self._capital_price(state, control)
        gap_sq = (investment / capital) ** 2 - self.params.delta**2
        gain = (1.0 - self.params.delta) + 0.5 * self.params.phi * gap_sq
        return (marginal * marginal_product + price * gain)[..., None]

    def residuals(self, state, control, expectation):
        price = self._capital_price(state, control)
        euler = price / (self.params.beta * expectation[..., 0]) - 1.0
        return euler[..., None]

    def _capital_price(self, state, control):
        capital, _, investment, consumption, _ = self._allocation(state, control)
        marginal = self._marginal_utility(consumption)
        denom = 1.0 - self.params.phi * (investment / capital - self.params.delta)
        return marginal / denom

    def _allocation(self, state, control):
        capital = jnp.exp(state[..., 0])
        productivity = state[..., 1]
        output = jnp.exp(productivity) * capital ** self.params.alpha
        investment = control[..., 0] * output
        return capital, productivity, investment, output - investment, output

    def _excess(self, consumption):
        floor = jnp.asarray(1e-8, dtype=consumption.dtype)
        return jnp.maximum(consumption - self.c_bar, floor)

    def _marginal_utility(self, consumption):
        return self._excess(consumption) ** (-self.params.sigma)


def _steady_state(params):
    rental = 1.0 / params.beta - 1.0 + params.delta
    capital = (params.alpha / rental) ** (1.0 / (1.0 - params.alpha))
    output = capital ** params.alpha
    investment = params.delta * capital
    consumption = output - investment
    c_bar = params.cbar_frac * consumption
    state = jnp.stack([jnp.log(capital), jnp.zeros_like(capital)])
    control = jnp.stack([investment / output])
    return SteadyState(
        state=state,
        control=control,
        K=capital,
        Y=output,
        C=consumption,
        I=investment,
        c_bar=c_bar,
    )
