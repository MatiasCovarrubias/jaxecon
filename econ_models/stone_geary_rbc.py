"""One-sector Stone-Geary RBC in the shared model contract."""

from typing import NamedTuple

from jax import numpy as jnp
from jax import random


class Params(NamedTuple):
    beta: jnp.ndarray
    alpha: jnp.ndarray
    delta: jnp.ndarray
    sigma: jnp.ndarray
    rho: jnp.ndarray
    shock_sd: jnp.ndarray
    cbar_frac: jnp.ndarray


class SteadyState(NamedTuple):
    state: jnp.ndarray
    control: jnp.ndarray
    K: jnp.ndarray
    Y: jnp.ndarray
    C: jnp.ndarray
    I: jnp.ndarray
    c_bar: jnp.ndarray


class StoneGearyRbc:
    """Planner RBC with subsistence consumption ``c_bar = cbar_frac * C_ss``.

    The state is ``[log K, log TFP]`` and the control is the saving rate.
    ``state_sd`` and ``control_sd`` are ones, so a caller may normalize by
    subtracting the steady state. ``expectation`` is the realized Euler
    continuation, and ``residuals`` is the relative Euler equation.
    """

    n_endogenous = 1

    def __init__(
        self,
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        sigma=2.0,
        rho=0.7,
        shock_sd=0.02,
        cbar_frac=0.1,
        precision=jnp.float32,
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

        dtype = precision
        self.params = Params(
            beta=jnp.asarray(beta, dtype=dtype),
            alpha=jnp.asarray(alpha, dtype=dtype),
            delta=jnp.asarray(delta, dtype=dtype),
            sigma=jnp.asarray(sigma, dtype=dtype),
            rho=jnp.asarray(rho, dtype=dtype),
            shock_sd=jnp.asarray(shock_sd, dtype=dtype),
            cbar_frac=jnp.asarray(cbar_frac, dtype=dtype),
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
        width = jnp.array([0.01, self.params.shock_sd], dtype=self.state_ss.dtype)
        return self.state_ss + width * random.normal(rng, self.state_ss.shape, dtype=self.state_ss.dtype)

    def sample_shock(self, rng):
        return random.normal(rng, (1,), dtype=self.state_ss.dtype)

    def transition(self, state, control, shock):
        capital, productivity, investment, _ = self._allocation(state, control)
        capital_next = (1.0 - self.params.delta) * capital + investment
        productivity_next = self.params.rho * productivity + self.params.shock_sd * shock[..., 0]
        return jnp.stack([jnp.log(capital_next), productivity_next], axis=-1)

    def utility(self, state, control):
        *_, consumption = self._allocation(state, control)
        excess = consumption - self.c_bar
        sigma = self.params.sigma
        return excess ** (1.0 - sigma) / (1.0 - sigma)

    def expectation(self, state, control):
        capital, productivity, _, consumption = self._allocation(state, control)
        marginal = self._marginal_utility(consumption)
        marginal_product = self.params.alpha * jnp.exp(productivity) * capital ** (self.params.alpha - 1.0)
        return (marginal * (marginal_product + 1.0 - self.params.delta))[..., None]

    def residuals(self, state, control, expectation):
        *_, consumption = self._allocation(state, control)
        marginal = self._marginal_utility(consumption)
        euler = (marginal - self.params.beta * expectation[..., 0]) / marginal
        return euler[..., None]

    def _allocation(self, state, control):
        capital = jnp.exp(state[..., 0])
        productivity = state[..., 1]
        output = jnp.exp(productivity) * capital ** self.params.alpha
        investment = control[..., 0] * output
        return capital, productivity, investment, output - investment

    def _marginal_utility(self, consumption):
        return (consumption - self.c_bar) ** (-self.params.sigma)


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
