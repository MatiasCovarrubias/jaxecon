"""Layer 0: one-sector RBC, optional irreversible investment."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from TimeIteration.models.processes import rouwenhorst
from TimeIteration.models.protocol import GridSpec, Grids


class Params(NamedTuple):
    beta: Array
    alpha: Array
    delta: Array
    sigma: Array
    rho: Array
    shock_sd: Array
    i_min_frac: Array
    phi: Array
    cbar_frac: Array


class SteadyState(NamedTuple):
    K: Array
    I: Array
    C: Array
    Y: Array
    u_c: Array
    s: Array


class Auxiliary(NamedTuple):
    C: Array
    Y: Array
    I: Array
    mu: Array
    A: Array
    u_c: Array
    s: Array


def _asarray(value) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


def default_params(**overrides) -> Params:
    values = dict(
        beta=0.99,
        alpha=0.3,
        delta=0.05,
        sigma=2.0,
        rho=0.7,
        shock_sd=0.02,
        i_min_frac=0.0,
        phi=0.0,
        cbar_frac=0.0,
    )
    values.update(overrides)
    return Params(**{key: _asarray(val) for key, val in values.items()})


def closed_form_params(**overrides) -> Params:
    values = dict(delta=1.0, sigma=1.0, i_min_frac=0.0, phi=0.0)
    values.update(overrides)
    return default_params(**values)


def subsistence_level(params: Params, C_ss: Array) -> Array:
    return params.cbar_frac * C_ss


def excess_consumption(C: Array, c_bar: Array) -> Array:
    return jnp.maximum(C - c_bar, jnp.array(1e-12, dtype=C.dtype))


def marginal_utility(C: Array, sigma: Array, c_bar: Array | float = 0.0) -> Array:
    excess = excess_consumption(C, jnp.asarray(c_bar, dtype=C.dtype))
    return excess ** (-sigma)


def period_utility(C: Array, sigma: Array, c_bar: Array | float = 0.0) -> Array:
    excess = excess_consumption(C, jnp.asarray(c_bar, dtype=C.dtype))
    crra = excess ** (1.0 - sigma) / (1.0 - sigma)
    return jnp.where(jnp.abs(sigma - 1.0) < 1e-12, jnp.log(excess), crra)


class RbcModel:
    """RBC residual. `x` is `(I,)` or `(I, ?)`; set `unknown='saving_rate'` to store `s`."""

    def __init__(self, irreversible: bool = False, unknown: str = "investment"):
        if unknown not in ("investment", "saving_rate"):
            raise ValueError("unknown must be 'investment' or 'saving_rate'")
        self.irreversible = bool(irreversible)
        self.unknown = unknown
        self.n_x = 2 if self.irreversible else 1

    def _investment(self, x: Array) -> Array:
        return x[..., 0]

    def _multiplier(self, x: Array) -> Array:
        if self.n_x > 1:
            return x[..., 1]
        return jnp.zeros_like(x[..., 0])

    def production(self, params: Params, a: Array, K: Array) -> tuple[Array, Array]:
        A = jnp.exp(a)
        return A, A * K**params.alpha

    def investment_from_unknown(self, params: Params, a: Array, K: Array, x: Array) -> Array:
        if self.unknown == "saving_rate":
            _, Y = self.production(params, a, K)
            return self._investment(x) * Y
        return self._investment(x)

    def steady_state(self, params: Params) -> SteadyState:
        rental = 1.0 / params.beta - 1.0 + params.delta
        K = (params.alpha / rental) ** (1.0 / (1.0 - params.alpha))
        Y = K**params.alpha
        I = params.delta * K
        C = Y - I
        c_bar = subsistence_level(params, C)
        return SteadyState(K=K, I=I, C=C, Y=Y, u_c=marginal_utility(C, params.sigma, c_bar), s=I / Y)

    def exog_process(self, params: Params, spec: GridSpec) -> tuple[Array, Array]:
        return rouwenhorst(spec.n_a, params.rho, params.shock_sd)

    def endo_grid(self, params: Params, ss: SteadyState, spec: GridSpec) -> Array:
        del params
        return ss.K * jnp.geomspace(spec.k_min_rel, spec.k_max_rel, spec.n_k)

    def transition(self, params: Params, a: Array, K: Array, x: Array) -> Array:
        I = self.investment_from_unknown(params, a, K, x)
        adjustment = 0.5 * params.phi * (I / K - params.delta) ** 2 * K
        return (1.0 - params.delta) * K + I - adjustment

    def auxiliary(self, params: Params, a: Array, K: Array, x: Array, ss: SteadyState) -> Auxiliary:
        A, Y = self.production(params, a, K)
        I = self.investment_from_unknown(params, a, K, x)
        C = Y - I
        c_bar = subsistence_level(params, ss.C)
        return Auxiliary(
            C=C,
            Y=Y,
            I=I,
            mu=self._multiplier(x),
            A=A,
            u_c=marginal_utility(C, params.sigma, c_bar),
            s=I / Y,
        )

    def capital_price(self, params: Params, K: Array, I: Array, u_c: Array) -> Array:
        return u_c / (1.0 - params.phi * (I / K - params.delta))

    def expectand(self, params: Params, a: Array, K: Array, x: Array, ss: SteadyState) -> Array:
        aux = self.auxiliary(params, a, K, x, ss)
        mpk = params.alpha * aux.A * K ** (params.alpha - 1.0)
        price = self.capital_price(params, K, aux.I, aux.u_c)
        continuation = (1.0 - params.delta) + 0.5 * params.phi * (aux.I**2 / K**2 - params.delta**2)
        return aux.u_c * mpk + price * continuation - (1.0 - params.delta) * aux.mu

    def arbitrage(self, params: Params, a: Array, K: Array, x: Array, Ex: Array, ss: SteadyState) -> Array:
        aux = self.auxiliary(params, a, K, x, ss)
        price = self.capital_price(params, K, aux.I, aux.u_c)
        mu_n = aux.mu / aux.u_c
        euler = (price * (1.0 - mu_n) - params.beta * Ex) / aux.u_c
        if not self.irreversible:
            return euler[None]
        slack = (aux.I - params.i_min_frac * ss.I) / ss.I
        complementarity = jnp.minimum(slack, aux.mu / ss.u_c)
        return jnp.stack([euler, complementarity])

    def initial_policy(self, params: Params, grids: Grids) -> Array:
        a = grids.a_nodes[:, None]
        K = grids.K_grid[None, :]
        if self.unknown == "saving_rate":
            control = jnp.broadcast_to(grids.ss.s, (a.shape[0], K.shape[1]))
        else:
            control = jnp.broadcast_to(grids.ss.I, (a.shape[0], K.shape[1]))
        x0 = control[..., None]
        if self.n_x == 1:
            return x0
        return jnp.concatenate([x0, jnp.zeros_like(x0)], axis=-1)
