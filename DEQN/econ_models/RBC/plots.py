"""Eval-only diagnostics for the shared RBC DEQN vs APG comparison."""

import json
import math
import os

import matplotlib.pyplot as plt


def load_welfare_history(results_path):
    if not results_path or not os.path.isfile(results_path):
        return []
    with open(results_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("welfare_history", [])


def collect_eval_histories(own_label, own_history, peer_label, peer_results_path):
    histories = {}
    if own_history:
        histories[own_label] = own_history
    peer_history = load_welfare_history(peer_results_path)
    if peer_history:
        histories[peer_label] = peer_history
    return histories


def _epochs(history):
    return [row["epoch"] for row in history]


def _values(history, key, scale=1.0):
    return [scale * row[key] for row in history if key in row]


def _has_key(history, key):
    return bool(history) and key in history[0]


def plot_eval_comparison(histories, save_path, steps_per_epoch=5, display_dpi=100):
    """Overlay eval welfare and volatilities for each algorithm.

    `histories` maps a label (e.g. "DEQN") to a welfare_history list.
    """
    if not histories:
        return None

    has_vols = all(_has_key(rows, "std_y") for rows in histories.values())
    has_euler = all(_has_key(rows, "euler_acc") for rows in histories.values())
    panels = ["ce", "k_rel"]
    if has_vols:
        panels.extend(["std_c", "std_y", "std_i", "i_y"])
    if has_euler:
        panels.append("euler")

    ncols = 2 if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.0 * nrows), squeeze=False)
    flat = axes.flat
    xlabel = "Adam steps"

    for i, name in enumerate(panels):
        ax = flat[i]
        for label, rows in histories.items():
            steps = [epoch * steps_per_epoch for epoch in _epochs(rows)]
            if name == "ce":
                ax.plot(steps, _values(rows, "ce_vs_fixed_saving_rate", 100.0), label=label)
            elif name == "k_rel":
                ax.plot(steps, _values(rows, "K_rel"), label=label)
            elif name == "std_c":
                ax.plot(steps, _values(rows, "std_c"), label=label)
            elif name == "std_y":
                ax.plot(steps, _values(rows, "std_y"), label=label)
            elif name == "std_i":
                ax.plot(steps, _values(rows, "std_i"), label=label)
            elif name == "i_y":
                ratios = [
                    row["std_i"] / row["std_y"] if row.get("std_y") else float("nan")
                    for row in rows
                ]
                ax.plot(steps, ratios, label=label)
            elif name == "euler":
                ax.plot(steps, _values(rows, "euler_acc", 100.0), label=label)

        if name == "ce":
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.set_ylabel("CE vs $s_{ss}$ (%)")
            ax.set_title("Consumption equivalent vs $s_{ss}$")
        elif name == "k_rel":
            ax.axhline(1.0, color="0.5", linewidth=0.8)
            ax.set_ylabel(r"$K/K_{ss}$")
            ax.set_title("Mean capital")
        elif name == "std_c":
            ax.set_ylabel(r"$\sigma(\log C)$")
            ax.set_title("Consumption volatility")
        elif name == "std_y":
            ax.set_ylabel(r"$\sigma(\log Y)$")
            ax.set_title("GDP volatility")
        elif name == "std_i":
            ax.set_ylabel(r"$\sigma(\log I)$")
            ax.set_title("Investment volatility")
        elif name == "i_y":
            ax.axhline(1.0, color="0.5", linewidth=0.8)
            ax.set_ylabel(r"$\sigma(\log I)/\sigma(\log Y)$")
            ax.set_title("Relative investment volatility")
        elif name == "euler":
            ax.set_ylabel("Euler accuracy (%)")
            ax.set_title("Euler accuracy")
        ax.set_xlabel(xlabel)
        if len(histories) > 1:
            ax.legend()

    for j in range(len(panels), nrows * ncols):
        flat[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if plt.isinteractive():
        plt.show()
    return fig
