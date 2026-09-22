"""Plots and tables for one or more policies on a shared model.

``policies`` maps a name to ``policy(state) -> control``. A neural net and a
time-iteration grid are the same kind of object. Every comparison uses one
common draw of initial states and shocks.

States are ``[log K, log TFP]``. Log deviations are differences from the
deterministic steady state. The saving-rate log deviation is
``log s - log s_ss``. Policy slices plot the saving rate in levels, on
``K/K_ss`` and on ``log A``.
"""

from typing import NamedTuple

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from trainers.evaluate import draw_sample, path

COLORS = ("#001C7F", "#B1400D", "#12711C", "#6C2C91")
LINE_STYLES = ("-", "--", ":", "-.")

_DEFAULTS = {
    "episodes": 32,
    "periods": 256,
    "burn": 64,
    "seed": 1,
    "trajectory_periods": 80,
    "irf_periods": 40,
    "irf_size": 1.0,
    "ss_horizon": 200,
    "ss_starts": 16,
    "policy_nsd": 2.0,
    "policy_k_min": 0.2,
    "policy_k_max": 2.0,
}


class Sample(NamedTuple):
    """Shared panel behind the histograms, moments, and trajectory.

    ``state`` and ``control`` are log deviations after the burn-in, shaped
    ``(episodes, kept periods, dim)``. ``path_state`` and ``path_control`` are
    the first episode in the model's own coordinates, from the initial state.
    """

    names: tuple
    state: dict
    control: dict
    path_state: dict
    path_control: dict


class Report(NamedTuple):
    """Text tables and the four figures. Figures are closed when saved."""

    moments: str
    stochastic_steady_state: str
    figures: dict


def simulate_policies(model, policies, config=None):
    """Roll every policy on one shared sample of states and shocks."""
    settings = dict(_DEFAULTS)
    if config:
        settings.update(config)
    names = tuple(policies)
    if not names:
        raise ValueError("policies must name at least one policy")
    episodes = int(settings["episodes"])
    periods = int(settings["periods"])
    burn = int(settings["burn"])
    if burn < 0 or burn >= periods:
        raise ValueError("burn must be shorter than the simulated path")
    show = min(int(settings["trajectory_periods"]), periods)
    states0, shocks, _ = draw_sample(model, episodes, periods, 1, settings["seed"])

    state = {}
    control = {}
    path_state = {}
    path_control = {}
    for name, policy in policies.items():
        levels, choices = _roll(model, policy, states0, shocks)
        state[name] = levels[:, burn:] - model.state_ss
        control[name] = _log_control(choices[:, burn:], model.control_ss)
        path_state[name] = levels[0, :show]
        path_control[name] = choices[0, :show]
    return Sample(names, state, control, path_state, path_control)


def moment_table(sample):
    """Mean, sample sd, skewness, and excess kurtosis of the log deviations.

    Mean and sd are in percent. Skewness and excess kurtosis are the raw
    population moment ratios.
    """
    header = f"{'solution':<16} {'variable':<18} {'mean_%':>10} {'sd_%':>10} {'skew':>10} {'exkurt':>10}"
    lines = [header]
    for name in sample.names:
        for label, values in _series(sample, name).items():
            mean, sd, skew, exkurt = _moments(values)
            lines.append(
                f"{name:<16} {label:<18} {_num(100 * mean):>10} {_num(100 * sd):>10} "
                f"{_num(skew):>10} {_num(exkurt):>10}"
            )
    return "\n".join(lines)


def stochastic_steady_state(model, policies, sample, config=None):
    """Zero-shock endpoints from each policy's own simulated states.

    The reported state is the mean endpoint. The saving rate is the policy at
    that state. ``endpoint_sd_%`` is the dispersion of those endpoints; a
    small number means the starts agreed.
    """
    settings = dict(_DEFAULTS)
    if config:
        settings.update(config)
    horizon = int(settings["ss_horizon"])
    n_starts = int(settings["ss_starts"])
    rows = {}
    dispersion = {}
    for name, policy in policies.items():
        starts = _starts(sample.state[name] + model.state_ss, n_starts)
        endpoints = _zero_shock_endpoints(model, policy, starts, horizon)
        mean_state = jnp.mean(endpoints, axis=0)
        choice = policy(mean_state)
        rows[name] = {
            "log K": mean_state[0] - model.state_ss[0],
            "log TFP": mean_state[1] - model.state_ss[1],
            "log saving rate": jnp.log(choice[0]) - jnp.log(model.control_ss[0]),
        }
        spread = jnp.std(endpoints, axis=0)
        dispersion[name] = {"log K": spread[0], "log TFP": spread[1]}
    return _steady_table(sample.names, rows, dispersion)


def plot_policies(model, policies, sample=None, config=None):
    """Saving rate against capital at ``a=0``, and against TFP at ``K_ss``.

    Capital runs from ``policy_k_min`` to ``policy_k_max`` times steady-state
    capital, so the slice includes the low-capital bend. TFP uses
    ``policy_nsd`` unconditional AR(1) standard deviations. ``sample`` is
    accepted so a report can pass the shared panel through.
    """
    del sample
    settings = dict(_DEFAULTS)
    if config:
        settings.update(config)
    nsd = float(settings["policy_nsd"])
    k_rel = jnp.linspace(
        jnp.asarray(settings["policy_k_min"], dtype=model.state_ss.dtype),
        jnp.asarray(settings["policy_k_max"], dtype=model.state_ss.dtype),
        161,
    )
    a_sd = model.params.shock_sd / jnp.sqrt(1.0 - model.params.rho**2)
    a_grid = jnp.linspace(-nsd * a_sd, nsd * a_sd, 81)
    curves = {
        name: _policy_curves(model, policy, k_rel, a_grid) for name, policy in policies.items()
    }

    with _style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for index, (name, (along_k, along_a)) in enumerate(curves.items()):
            color, linestyle = _line(index)
            axes[0].plot(np.asarray(k_rel), np.asarray(along_k), color=color, linestyle=linestyle, label=name)
            axes[1].plot(np.asarray(a_grid), np.asarray(along_a), color=color, linestyle=linestyle, label=name)
        s_ss = float(model.control_ss[0])
        for ax, xlabel, title in (
            (axes[0], r"$K/K_{\mathrm{ss}}$", r"Policy vs capital ($a=0$)"),
            (axes[1], r"$\log A$", r"Policy vs TFP ($K=K_{\mathrm{ss}}$)"),
        ):
            ax.axhline(s_ss, color="0.45", linewidth=1.0, linestyle=":", label=r"$s_{\mathrm{ss}}$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Saving rate")
            ax.set_title(title)
            ax.legend()
            _format_axis(ax)
        fig.tight_layout()
    return fig


def plot_trajectory(model, sample):
    """First shared path. States are log deviations; the saving rate is a level."""
    panels = (
        (lambda state: state[:, 0] - float(model.state_ss[0]), r"$\log K - \log K_{\mathrm{ss}}$"),
        (lambda state: state[:, 1] - float(model.state_ss[1]), r"$\log A$"),
    )
    with _style():
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
        for index, name in enumerate(sample.names):
            color, linestyle = _line(index)
            state = np.asarray(sample.path_state[name])
            saving = np.asarray(sample.path_control[name][:, 0])
            time = np.arange(state.shape[0])
            for ax, (series, _) in zip(axes, panels):
                ax.plot(time, series(state), color=color, linestyle=linestyle, label=name)
            axes[2].plot(time, saving, color=color, linestyle=linestyle, label=name)
        for ax, (_, ylabel) in zip(axes, panels):
            ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle=":")
            ax.set_ylabel(ylabel)
            ax.legend()
            _format_axis(ax)
        axes[2].axhline(float(model.control_ss[0]), color="0.45", linewidth=1.0, linestyle=":")
        axes[2].set_ylabel("Saving rate")
        axes[2].set_xlabel("Period")
        axes[2].legend()
        _format_axis(axes[2])
        fig.tight_layout()
    return fig


def plot_histograms(sample):
    """Ergodic log deviations, one panel per variable, policies overlaid."""
    labels = tuple(_series(sample, sample.names[0]))
    with _style():
        fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.8), squeeze=False)
        flat_axes = axes[0]
        for ax, label in zip(flat_axes, labels):
            columns = [np.asarray(_series(sample, name)[label], dtype=float).ravel() for name in sample.names]
            pooled = np.concatenate([column[np.isfinite(column)] for column in columns])
            lo, hi = float(np.min(pooled)), float(np.max(pooled))
            if hi - lo < 1e-6:
                lo, hi = lo - 1e-3, hi + 1e-3
            bins = np.linspace(lo, hi, 31)
            for index, (name, column) in enumerate(zip(sample.names, columns)):
                color, _ = _line(index)
                ax.hist(
                    column[np.isfinite(column)],
                    bins=bins,
                    density=True,
                    alpha=0.35,
                    color=color,
                    label=name,
                )
            ax.axvline(0.0, color="0.45", linewidth=1.0, linestyle=":")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.legend()
            _format_axis(ax)
        fig.tight_layout()
    return fig


def plot_impulses(model, policies, config=None):
    """One-standard-deviation innovation at date 0, then zeros.

    Date 0 is the deterministic steady state, before the shock. The innovation
    is ``1``, and the model scales it by ``shock_sd``.
    """
    settings = dict(_DEFAULTS)
    if config:
        settings.update(config)
    periods = int(settings["irf_periods"])
    size = float(settings["irf_size"])
    responses = {name: _impulse(model, policy, periods, size) for name, policy in policies.items()}
    panels = (
        (0, r"$\log K - \log K_{\mathrm{ss}}$"),
        (1, r"$\log A$"),
        (2, r"$\log s - \log s_{\mathrm{ss}}$"),
    )
    with _style():
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
        time = np.arange(periods)
        for index, name in enumerate(policies):
            color, linestyle = _line(index)
            state, saving = responses[name]
            series = (np.asarray(state[:, 0]), np.asarray(state[:, 1]), np.asarray(saving))
            for ax, values in zip(axes, series):
                ax.plot(time, values, color=color, linestyle=linestyle, label=name)
        for ax, (_, ylabel) in zip(axes, panels):
            ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle=":")
            ax.set_ylabel(ylabel)
            ax.legend()
            _format_axis(ax)
        axes[-1].set_xlabel("Period")
        fig.tight_layout()
    return fig


def report(model, policies, directory=None, config=None):
    """Build the sample, the two tables, and the four figures.

    With ``directory``, write ``moments.txt``, ``stochastic_steady_state.txt``,
    and each figure as ``.pdf`` and ``.png``.
    """
    sample = simulate_policies(model, policies, config)
    moments = moment_table(sample)
    steady = stochastic_steady_state(model, policies, sample, config)
    figures = {
        "policy": plot_policies(model, policies, sample, config),
        "trajectory": plot_trajectory(model, sample),
        "ergodic": plot_histograms(sample),
        "impulse": plot_impulses(model, policies, config),
    }
    print(moments, flush=True)
    print(flush=True)
    print(steady, flush=True)
    if directory is not None:
        from pathlib import Path

        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "moments.txt").write_text(moments + "\n", encoding="utf-8")
        (folder / "stochastic_steady_state.txt").write_text(steady + "\n", encoding="utf-8")
        for stem, fig in figures.items():
            _save(fig, folder, stem)
        figures = {}
    return Report(moments, steady, figures)


def _roll(model, policy, states0, shocks):
    def episode(state, episode_shocks):
        _, states, _ = path(model, policy, state, episode_shocks)
        return states, jax.vmap(policy)(states)

    return jax.jit(jax.vmap(episode))(states0, shocks)


def _policy_curves(model, policy, k_rel, a_grid):
    logk = model.state_ss[0] + jnp.log(k_rel)
    capital_states = jnp.stack([logk, jnp.zeros_like(logk)], axis=-1)
    productivity_states = jnp.stack([jnp.full_like(a_grid, model.state_ss[0]), a_grid], axis=-1)

    def column(states):
        return jax.vmap(policy)(states)[:, 0]

    return jax.jit(column)(capital_states), jax.jit(column)(productivity_states)


def _impulse(model, policy, periods, size):
    shock_dim = int(model.sample_shock(jax.random.PRNGKey(0)).shape[0])
    shocks = jnp.zeros((periods, shock_dim), dtype=model.state_ss.dtype).at[0, 0].set(size)

    def run(state0):
        _, states, _ = path(model, policy, state0, shocks)
        controls = jax.vmap(policy)(states)
        return states - model.state_ss, jnp.log(controls[:, 0]) - jnp.log(model.control_ss[0])

    return jax.jit(run)(model.state_ss)


def _zero_shock_endpoints(model, policy, starts, horizon):
    shock_dim = int(model.sample_shock(jax.random.PRNGKey(0)).shape[0])
    shocks = jnp.zeros((horizon, shock_dim), dtype=starts.dtype)

    def one(start):
        final, _, _ = path(model, policy, start, shocks)
        return final

    return jax.jit(jax.vmap(one))(starts)


def _starts(states, n_starts):
    flat = states.reshape((-1, states.shape[-1]))
    count = min(int(n_starts), int(flat.shape[0]))
    index = jnp.linspace(0, flat.shape[0] - 1, count).astype(jnp.int32)
    return flat[index]


def _series(sample, name):
    state = sample.state[name]
    control = sample.control[name]
    return {
        "log K": state[..., 0],
        "log TFP": state[..., 1],
        "log saving rate": control[..., 0],
    }


def _log_control(control, control_ss):
    return jnp.log(control) - jnp.log(control_ss)


def _num(value):
    if not np.isfinite(value):
        return ""
    return f"{float(value):.4f}"


def _moments(values):
    data = np.asarray(values, dtype=float).ravel()
    data = data[np.isfinite(data)]
    count = data.size
    if count < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    mean = float(data.mean())
    centered = data - mean
    second = float(np.mean(centered**2))
    sd = float(np.sqrt(np.sum(centered**2) / (count - 1)))
    if second <= 0.0:
        return (mean, sd, np.nan, np.nan)
    skew = float(np.mean(centered**3) / second**1.5)
    exkurt = float(np.mean(centered**4) / second**2 - 3.0)
    return (mean, sd, skew, exkurt)


def _steady_table(names, rows, dispersion):
    header = f"{'solution':<16} {'variable':<18} {'logdev_%':>10} {'endpoint_sd_%':>14}"
    lines = [header]
    for name in names:
        for label, value in rows[name].items():
            spread = dispersion[name].get(label)
            spread_text = f"{100 * float(spread):14.4f}" if spread is not None else f"{'':>14}"
            lines.append(f"{name:<16} {label:<18} {100 * float(value):10.4f} {spread_text}")
    return "\n".join(lines)


def _line(index):
    return COLORS[index % len(COLORS)], LINE_STYLES[index % len(LINE_STYLES)]


def _format_axis(ax):
    ax.grid(True, color="#B0B0B0", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _style():
    return plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "lines.linewidth": 2.0,
            "figure.dpi": 120,
        }
    )


def _save(fig, folder, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"{stem}.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)
