"""Environment interface for the APG rollout, and a checker for it."""

from abc import ABC, abstractmethod

import jax
import numpy as np
from jax import numpy as jnp
from jax import random


class WelfareEnvironment(ABC):
    """Interface required by the core APG rollout.

    Coordinates. ``state`` is the network input: deviations from the
    deterministic steady state divided by ``state_sd``, so the steady state is
    the zero vector. ``action`` is the network output: deviations from
    ``policies_ss`` in units of ``policies_sd``, so
    ``deterministic_steady_state_action()`` is the zero vector. The environment
    maps actions into economic quantities through bounded functions so that
    every flow stays strictly positive for any real-valued action.

    Attributes the rollout and trainers read:

    - ``obs_ss``: shape and dtype template for network initialization (values unused);
    - ``action_dim``; ``discount_rate``; ``value_ss`` (critic scale);
    - ``state_sd``, ``policies_sd`` and ``set_scales(state_sd, policies_sd)``,
      used by the log-linear baseline in ``APG.loglinear``;
    - ``precision``.

    Shapes. ``state`` is ``(..., state_dim)``, ``action`` is ``(..., action_dim)``,
    ``shock`` is ``(..., n_shocks)``; the rollout uses no leading dimension and
    needs a scalar reward. Run ``check_environment(env)`` on a new
    implementation, then ``APG.loglinear.solve.linearize_environment(env)`` in
    float64: a zero ``foc_residual`` confirms the steady state and the
    coordinate conventions. See the Environment Interface section of
    ``APG/README.md``.
    """

    @abstractmethod
    def initial_state(self, rng, init_range=0):
        """Draw an initial state.

        ``init_range`` is a percentage: levels are drawn uniformly within
        ``±init_range`` percent of their steady-state values; ``init_range=0``
        returns the deterministic steady state exactly (the zero vector).
        """

    @abstractmethod
    def sample_shock(self, rng):
        """Draw one period's standard-normal shock vector of shape ``(n_shocks,)``.

        The rollout multiplies it by ``simul_vol_scale``; the environment applies
        the standard deviations inside ``transition``.
        """

    @abstractmethod
    def transition(self, state, action, shock):
        """Return next period's normalized state. Differentiable in ``state`` and ``action``."""

    @abstractmethod
    def training_reward(self, state, action):
        """Return the period reward optimized during training, a scalar. Differentiable in ``action``."""

    @abstractmethod
    def deterministic_steady_state_action(self):
        """Return the action that implements the deterministic steady state: the zero vector."""

    @abstractmethod
    def terminal_value(self, state, horizon=512):
        """Return the discounted utility of steady-state actions for ``horizon`` periods from ``state``.

        Shocks are at their conditional mean (zero). Differentiable in ``state``.
        Used when ``use_model_terminal_value`` is on; other tails should be
        separate, named options.
        """

    @abstractmethod
    def deterministic_steady_state_welfare(self, horizon):
        """Return ``reward_ss * sum_{t<horizon} discount_rate**t``."""

    @abstractmethod
    def consumption_equivalent(self, welfare, baseline_welfare, horizon):
        """Return the fraction ``x`` such that scaling the baseline consumption path by ``1 + x`` yields ``welfare``.

        Depends only on the period utility; ``horizon`` is needed when the
        utility is not homogeneous in consumption (the logarithmic case).
        """


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _finite(value, name):
    _require(jnp.all(jnp.isfinite(value)), f"{name} is not finite: {value}")


def check_environment(env, rng=None, action_scale=1e3, horizon=16, rtol=1e-5, atol=1e-6):
    """Verify that ``env`` satisfies the APG environment contract.

    Raises ``AssertionError`` on the first violated property. The properties:

    1. ``initial_state(rng, 0)`` has the shape of ``obs_ss``; ``action_dim`` is a
       positive integer; ``discount_rate`` is in ``(0, 1)``; ``value_ss`` is finite.
    2. ``deterministic_steady_state_action()`` has ``action_dim`` entries and,
       applied at the steady state with a zero shock, returns the same state.
    3. ``training_reward`` at the steady state is a finite scalar.
    4. Reward and next state stay finite for actions of size ``±action_scale``.
    5. Reward and transition are differentiable in the action with finite
       gradients, and the transition gradient is not identically zero.
    6. ``terminal_value`` is finite and differentiable in the state.
    7. ``deterministic_steady_state_welfare(horizon)`` equals the steady-state
       reward times the discount sum, and ``consumption_equivalent`` of a welfare
       against itself is zero.

    Not checked here: that the steady state satisfies the planner's first-order
    conditions, that the zero state and zero action are the steady state in the
    network's coordinates, or that the resource constraint holds with equality.
    ``APG.loglinear.solve.linearize_environment(env)["foc_residual"]`` in
    float64 covers the first two; an accounting test on random states and
    actions covers the third (see ``tests/test_time_to_build.py``).
    """
    rng = random.PRNGKey(0) if rng is None else rng
    obs_ss = jnp.asarray(env.obs_ss)

    state = env.initial_state(rng, 0)
    _require(state.shape == obs_ss.shape, f"initial_state shape {state.shape} != obs_ss shape {obs_ss.shape}")
    action_dim = int(env.action_dim)
    _require(action_dim > 0, "action_dim must be positive")
    discount = float(env.discount_rate)
    _require(0 < discount < 1, f"discount_rate must be in (0, 1), got {discount}")
    _finite(jnp.asarray(env.value_ss), "value_ss")

    action_ss = jnp.asarray(env.deterministic_steady_state_action())
    _require(action_ss.shape[-1] == action_dim, f"steady-state action has {action_ss.shape[-1]} entries, expected {action_dim}")
    zero_shock = jnp.zeros_like(env.sample_shock(rng))
    next_state = env.transition(state, action_ss, zero_shock)
    np.testing.assert_allclose(
        next_state, state, rtol=rtol, atol=atol, err_msg="steady-state action is not a fixed point under zero shock"
    )

    reward_ss = env.training_reward(state, action_ss)
    _require(jnp.ndim(reward_ss) == 0, f"training_reward must be scalar, got shape {jnp.shape(reward_ss)}")
    _finite(reward_ss, "steady-state training_reward")

    for sign in (-1.0, 1.0):
        extreme = jnp.full_like(action_ss, sign * action_scale)
        _finite(env.training_reward(state, extreme), f"training_reward at action {sign * action_scale}")
        _finite(env.transition(state, extreme, zero_shock), f"transition at action {sign * action_scale}")

    reward_grad = jax.grad(lambda a: env.training_reward(state, a))(action_ss)
    _finite(reward_grad, "d training_reward / d action")
    transition_jac = jax.jacobian(lambda a: env.transition(state, a, zero_shock))(action_ss)
    _finite(transition_jac, "d transition / d action")
    _require(jnp.any(transition_jac != 0), "transition does not depend on the action")

    shocked_state = env.transition(state, action_ss, jnp.ones_like(zero_shock))
    tail = env.terminal_value(shocked_state, horizon)
    _finite(tail, "terminal_value")
    tail_grad = jax.grad(lambda s: env.terminal_value(s, horizon))(shocked_state)
    _finite(tail_grad, "d terminal_value / d state")

    weight = sum(discount**t for t in range(horizon))
    np.testing.assert_allclose(
        env.deterministic_steady_state_welfare(horizon),
        reward_ss * weight,
        rtol=1e-4,
        err_msg="deterministic_steady_state_welfare != reward_ss * discount sum",
    )
    welfare_ss = env.deterministic_steady_state_welfare(horizon)
    np.testing.assert_allclose(
        env.consumption_equivalent(welfare_ss, welfare_ss, horizon),
        0.0,
        atol=atol,
        err_msg="consumption_equivalent of a welfare against itself is not zero",
    )
