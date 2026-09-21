from jax import lax, random, vmap
from jax import numpy as jnp

STATE_SAMPLING_MODES = ("occupancy", "grid", "mixed")


def sample_episode_shocks(econ_model, epis_rng, n_periods, vol_scale=1.0):
    """Draw one length-`n_periods` shock path from `epis_rng`.

    Uses the same key as `initial_state(epis_rng, ...)` so DEQN and APG keep
    a shared (initial state, shock path) pair. The antithetic counterpart is
    exactly `-shocks`.
    """
    period_rngs = random.split(epis_rng, n_periods)
    return vol_scale * vmap(econ_model.sample_shock)(jnp.stack(period_rngs))


def rollout_episode_obs(econ_model, train_state, init_obs, shocks):
    """Roll the policy from `init_obs` along a pre-drawn shock path."""

    def period_step(env_obs, shock):
        policy = train_state.apply_fn(train_state.params, env_obs)
        obs_next = econ_model.step(env_obs, policy, shock)
        return obs_next, obs_next

    _, epis_obs = lax.scan(period_step, init_obs, shocks)
    return epis_obs


def create_episode_simul_fn(econ_model, config):
    """Create a function that simulates an episode of the environment.

    Returns observations of shape `(T, state_dim)`, or `(2T, state_dim)` when
    `antithetic_episodes` is on: the drawn shock path followed by its negation,
    both from the same initial state.
    """
    n_periods = config["periods_per_epis"]
    init_range = config.get("init_range", 0)
    init_range_a = config.get("init_range_a")
    vol_scale = config.get("simul_vol_scale", 1.0)
    antithetic = bool(config.get("antithetic_episodes", False))

    def sample_epis_obs(train_state, epis_rng):
        mode = config.get("initial_state_mode")
        kwargs = {}
        if init_range_a is not None:
            kwargs["init_range_a"] = init_range_a
        if mode is not None:
            kwargs["mode"] = mode
        init_obs = econ_model.initial_state(epis_rng, init_range, **kwargs)
        shocks = sample_episode_shocks(econ_model, epis_rng, n_periods, vol_scale)
        obs_plus = rollout_episode_obs(econ_model, train_state, init_obs, shocks)
        if not antithetic:
            return obs_plus
        obs_minus = rollout_episode_obs(econ_model, train_state, init_obs, -shocks)
        return jnp.concatenate([obs_plus, obs_minus], axis=0)

    return sample_epis_obs


def create_normalized_state_grid(econ_model, config):
    """Create a Cartesian grid in the model's normalized state coordinates."""
    sampling = config.get("state_sampling", {})
    lower = sampling.get("grid_min")
    upper = sampling.get("grid_max")
    points = sampling.get("grid_points")
    state_dim = int(econ_model.state_ss.shape[0])

    if lower is None or upper is None or points is None:
        raise ValueError("grid and mixed state sampling require grid_min, grid_max, and grid_points")
    if len(lower) != state_dim or len(upper) != state_dim or len(points) != state_dim:
        raise ValueError(f"grid bounds and points must each have state_dim={state_dim} entries")
    if any(int(n) < 2 for n in points):
        raise ValueError("every grid dimension must contain at least two points")
    if any(float(lo) >= float(hi) for lo, hi in zip(lower, upper)):
        raise ValueError("every grid_min entry must be smaller than grid_max")

    dtype = econ_model.state_ss.dtype
    include_zero = bool(sampling.get("grid_include_zero", True))
    axes = []
    for lo, hi, n in zip(lower, upper, points):
        axis = jnp.linspace(float(lo), float(hi), int(n), dtype=dtype)
        if include_zero and int(n) > 2 and float(lo) <= 0 <= float(hi):
            axis = axis.at[jnp.argmin(jnp.abs(axis))].set(0)
        axes.append(axis)
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([axis.reshape(-1) for axis in mesh], axis=-1)


def sample_balanced_grid(grid, sample_size, rng):
    """Sample a fixed-size batch while covering grid points evenly."""
    grid_size = int(grid.shape[0])
    repeats = (int(sample_size) + grid_size - 1) // grid_size
    indices = jnp.tile(jnp.arange(grid_size), repeats)
    indices = random.permutation(rng, indices)[:sample_size]
    return grid[indices]


def create_step_state_sampler(econ_model, config):
    """Create the current-state sampler used by one DEQN optimizer step."""
    sampling = config.get("state_sampling", {})
    mode = sampling.get("mode", "occupancy")
    if mode not in STATE_SAMPLING_MODES:
        raise ValueError(f"state_sampling.mode must be one of {STATE_SAMPLING_MODES}, got {mode!r}")

    episode_simul_fn = create_episode_simul_fn(econ_model, config)
    periods_per_step = int(config["periods_per_step"])
    state_dim = int(econ_model.state_ss.shape[0])

    def occupancy_states(train_state, rng):
        epis_rng = random.split(rng, config["epis_per_step"])
        states = vmap(episode_simul_fn, in_axes=(None, 0))(train_state, jnp.stack(epis_rng))
        return states.reshape(periods_per_step, state_dim)

    if mode == "occupancy":
        return occupancy_states

    grid = create_normalized_state_grid(econ_model, config)
    if mode == "grid":

        def grid_states(_train_state, rng):
            return sample_balanced_grid(grid, periods_per_step, rng)

        return grid_states

    grid_share = float(sampling.get("grid_share", 0.5))
    if not 0 < grid_share < 1:
        raise ValueError("mixed state sampling requires 0 < grid_share < 1")
    n_grid = int(round(periods_per_step * grid_share))
    n_occupancy = periods_per_step - n_grid

    def mixed_states(train_state, rng):
        occupancy_rng, grid_rng = random.split(rng)
        occupancy = occupancy_states(train_state, occupancy_rng)[:n_occupancy]
        grid_states = sample_balanced_grid(grid, n_grid, grid_rng)
        return jnp.concatenate([occupancy, grid_states], axis=0)

    return mixed_states
