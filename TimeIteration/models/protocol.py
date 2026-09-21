"""Layer 0 interface: the only per-model surface the solver calls."""

from typing import Any, NamedTuple, Protocol

from jax import Array


class GridSpec(NamedTuple):
    n_a: int = 7
    n_k: int = 51
    k_min_rel: float = 0.7
    k_max_rel: float = 1.4


class Grids(NamedTuple):
    a_nodes: Array
    P: Array
    K_grid: Array
    ss: Any


class TimeIterationModel(Protocol):
    n_x: int

    def steady_state(self, params: Any) -> Any: ...

    def exog_process(self, params: Any, spec: GridSpec) -> tuple[Array, Array]: ...

    def endo_grid(self, params: Any, ss: Any, spec: GridSpec) -> Array: ...

    def transition(self, params: Any, a: Array, K: Array, x: Array) -> Array: ...

    def auxiliary(self, params: Any, a: Array, K: Array, x: Array, ss: Any) -> Any: ...

    def expectand(self, params: Any, a: Array, K: Array, x: Array, ss: Any) -> Array: ...

    def arbitrage(self, params: Any, a: Array, K: Array, x: Array, Ex: Array, ss: Any) -> Array: ...

    def initial_policy(self, params: Any, grids: Grids) -> Array: ...
