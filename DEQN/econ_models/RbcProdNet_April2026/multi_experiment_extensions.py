#!/usr/bin/env python3
"""Run the minimal multi-experiment analysis used by the extensions table."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:

    def _colab_package_stack_is_usable() -> bool:
        try:
            import jax
            import numpy
            import scipy
            import scipy.io  # noqa: F401
        except Exception as exc:
            print(f"Package stack check failed in current kernel: {exc!r}")
            return False
        print("Package stack OK: " f"numpy={numpy.__version__} scipy={scipy.__version__} jax={jax.__version__}")
        return True

    repair_marker = "/content/.jaxecon_colab_numpy_repair_attempted"
    if _colab_package_stack_is_usable():
        if os.path.exists(repair_marker):
            os.remove(repair_marker)
    else:
        if os.path.exists(repair_marker):
            raise RuntimeError(
                "The Colab NumPy/SciPy/JAX stack is still inconsistent after a repair restart. "
                "Use Runtime > Disconnect and delete runtime, then rerun the cell."
            )
        print("Installing pinned NumPy/SciPy/JAX stack, then restarting Colab.")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "numpy==2.0.2",
                "scipy==1.15.3",
                "jax[cuda12]",
            ],
            check=True,
        )
        Path(repair_marker).write_text(
            "numpy==2.0.2 scipy==1.15.3 jax[cuda12]\n",
            encoding="utf-8",
        )
        print("Package stack repaired. Restarting Colab runtime; rerun the cell after reconnecting.")
        os.kill(os.getpid(), 9)

    print("Cloning jaxecon repository...")
    if not os.path.exists("/content/jaxecon"):
        subprocess.run(
            ["git", "clone", "https://github.com/MatiasCovarrubias/jaxecon"],
            check=True,
        )
    else:
        subprocess.run(
            ["git", "-C", "/content/jaxecon", "pull", "--ff-only"],
            check=True,
        )

    from google.colab import drive  # type: ignore

    print("Mounting Google Drive...")
    drive.mount("/content/drive")
    REPO_ROOT = Path("/content/jaxecon")
    DATA_ROOT = Path("/content/drive/MyDrive/Jaxecontemp")
else:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = REPO_ROOT / "DEQN" / "econ_models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
from jax import config as jax_config  # noqa: E402

from DEQN.analysis.model_hooks import load_model_analysis_hooks  # noqa: E402
from DEQN.analysis.simul_analysis import (  # noqa: E402
    create_episode_simulation_fn_verbose,
    simulation_analysis,
)
from DEQN.analysis.welfare import get_welfare_fn  # noqa: E402
from DEQN.analysis.welfare_outputs import (
    _compute_welfare_cost_from_sample,  # noqa: E402
)
from DEQN.econ_models import load_model_class  # noqa: E402
from DEQN.econ_models.RbcProdNet_April2026.plot_helpers import (  # noqa: E402
    _sectoral_levels_from_logdev,
    _sectoral_share_change,
    _sectoral_share_weights,
    _sectoral_variable_info,
)
from DEQN.training.checkpoints import (  # noqa: E402
    load_experiment_data,
    load_trained_model_orbax,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "model_dir": "RbcProdNet_April2026",
    "comparative_name": "extensions",
    "model_data_object": "ModelData",
    "double_precision": True,
    "init_range": 6,
    "periods_per_epis": 64000,
    "burn_in_periods": 3200,
    "simul_vol_scale": 1.0,
    "simul_seed": 0,
    "n_simul_seeds": 16,
    "welfare_n_trajects": 16000,
    "welfare_traject_length": 200,
    "welfare_seed": 0,
    "caption_tex": "Additional insights and robustness",
    "table_label": "tab:robustness_summary",
    "experiments": [],
}

# Edit this block when pasting the complete runner into a Colab cell.
config: dict[str, Any] = {
    "model_dir": "RbcProdNet_April2026",
    "comparative_name": "extensions",
    "caption_tex": "Additional insights and robustness",
    "table_label": "tab:robustness_summary",
    "experiments": [
        {
            "experiment_name": "finercal_July",
            "model_data_file": "ModelData_finercal.mat",
            "label": "Baseline",
            "section": "Baseline",
        },
        {
            "experiment_name": "homogenousshocks",
            "model_data_file": "ModelData_homogenousshocks.mat",
            "label": "Homogeneous shocks",
            "section": "Robustness",
        },
        {
            "experiment_name": "correlatedshocks",
            "model_data_file": "ModelData_correlatedshocks.mat",
            "label": "Correlated shocks",
            "section": "Robustness",
        },
        {
            "experiment_name": "sigmay0dot8",
            "model_data_file": "ModelData_sigmay0dot8.mat",
            "label_tex": r"High $\sigma_y$ (0.8)",
            "section": "Robustness",
        },
        {
            "experiment_name": "sigmam0dot1",
            "model_data_file": "ModelData_sigmam0dot1.mat",
            "label_tex": r"High $\sigma_m$ (0.1)",
            "section": "Robustness",
        },
        {
            "experiment_name": "CDdefault",
            "model_data_file": "ModelData_CDdefault.mat",
            "exact_cobb_douglas": False,
            "label_tex": r"Cobb--Douglas for $\sigma_c$, $\sigma_I$, $\sigma_q$",
            "section": "Robustness",
        },
        {
            "experiment_name": "CDexact",
            "model_data_file": "ModelData_CDexact.mat",
            "exact_cobb_douglas": True,
            "label_tex": "Cobb--Douglas for all elasticities",
            "section": "Robustness",
        },
    ],
    "notes_tex": (
        "Mean and standard deviation of consumption are moments of the ergodic distribution and "
        "are reported as percent log differences from the deterministic steady state. "
        "The welfare cost is the consumption-equivalent cost of business cycles, in percent. "
        "Capital reallocation is the sum of the absolute sectoral capital-share changes defined "
        "as in Figure \\ref{fig:cap_share_ergodic}. Each sectoral change is the percent change in its "
        "ergodic-mean capital share relative to its deterministic-steady-state share. "
        "``Homogeneous shocks'' sets sectoral shock volatilities and persistence parameters to common values."
    ),
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _escape_latex(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _display_text(spec: dict[str, Any], plain_key: str, tex_key: str, fallback: str) -> str:
    if spec.get(tex_key) is not None:
        return str(spec[tex_key])
    return _escape_latex(spec.get(plain_key, fallback))


def _validate_config(config: dict[str, Any]) -> None:
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("config['experiments'] must be a non-empty list.")

    names: list[str] = []
    for index, spec in enumerate(experiments):
        if not isinstance(spec, dict):
            raise ValueError(f"Experiment entry {index} must be a JSON object.")
        missing = [key for key in ("experiment_name", "model_data_file") if not spec.get(key)]
        if missing:
            raise ValueError(f"Experiment entry {index} is missing required keys: {missing}")
        names.append(str(spec["experiment_name"]))
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Experiment names must be unique: {duplicates}")

    if int(config["periods_per_epis"]) <= int(config["burn_in_periods"]):
        raise ValueError("periods_per_epis must exceed burn_in_periods.")


def _load_model_data(
    model_path: Path,
    *,
    object_name: str,
    precision: Any,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"ModelData file not found: {model_path}")
    loaded = sio.loadmat(model_path, simplify_cells=True)
    if object_name not in loaded:
        raise ValueError(f"Expected MATLAB object '{object_name}' in {model_path.name}.")

    model_data = loaded[object_name]
    steady_state = model_data["SteadyState"]
    parameters = steady_state["parameters"]
    n_sectors = int(parameters["parn_sectors"])
    state_ss = jnp.concatenate(
        [
            jnp.asarray(steady_state["endostates_ss"], dtype=precision),
            jnp.zeros((n_sectors,), dtype=precision),
        ]
    )
    policies_ss = jnp.asarray(steady_state["policies_ss"], dtype=precision)
    statistics = model_data["Statistics"]
    state_sd = jnp.asarray(statistics["states_sd"], dtype=precision)
    policies_sd = jnp.asarray(statistics["policies_sd"], dtype=precision)
    state_space = model_data["Solution"]["StateSpace"]
    policy_state_matrix = jnp.asarray(state_space["C"], dtype=precision)

    if policies_ss.shape[0] != policies_sd.shape[0]:
        n_policies = policies_sd.shape[0]
        policies_ss = policies_ss[:n_policies]
        policy_state_matrix = policy_state_matrix[:n_policies, :]

    return {
        "parameters": parameters,
        "n_sectors": n_sectors,
        "state_ss": state_ss,
        "policies_ss": policies_ss,
        "state_sd": state_sd,
        "policies_sd": policies_sd,
        "policy_state_matrix": policy_state_matrix,
    }


def _check_checkpoint_model_data(
    *,
    experiment_name: str,
    experiment_config: dict[str, Any],
    requested_model_data_file: str,
    allow_mismatch: bool,
) -> None:
    trained_file = experiment_config.get("model_data_file")
    if not trained_file:
        return
    if Path(str(trained_file)).name == Path(requested_model_data_file).name:
        return
    message = (
        f"Checkpoint '{experiment_name}' was trained with '{trained_file}', but this analysis "
        f"requested '{requested_model_data_file}'."
    )
    if not allow_mismatch:
        raise ValueError(message + " Set allow_model_data_mismatch=true only if this is intentional.")
    print(f"  Warning: {message}", flush=True)


def _ergodic_capital_reallocation_pp(
    *,
    simul_obs: Any,
    state_ss: Any,
    policies_ss: Any,
    n_sectors: int,
) -> float:
    variable_info = _sectoral_variable_info("K", n_sectors)
    capital_logdev = np.asarray(simul_obs, dtype=float)[:, :n_sectors]
    capital_ss_log = np.asarray(state_ss, dtype=float)[:n_sectors]
    capital_ergodic_mean = _sectoral_levels_from_logdev(capital_logdev, capital_ss_log)
    capital_deterministic_ss = np.exp(capital_ss_log)
    weights, _ = _sectoral_share_weights(policies_ss, variable_info, n_sectors)
    share_changes = _sectoral_share_change(
        capital_ergodic_mean,
        capital_deterministic_ss,
        weights,
    )
    if not np.all(np.isfinite(share_changes)):
        raise ValueError("Capital composition changes contain non-finite values.")
    return float(100.0 * np.sum(np.abs(share_changes)))


def _consumption_statistics(consumption_logdev: Any) -> tuple[float, float]:
    values = np.asarray(consumption_logdev, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("Aggregate consumption simulation has fewer than two finite observations.")
    return float(100.0 * np.mean(values)), float(100.0 * np.std(values, ddof=1))


def _run_experiment(
    *,
    config: dict[str, Any],
    spec: dict[str, Any],
    model_dir: Path,
    experiments_dir: Path,
    precision: Any,
) -> dict[str, Any]:
    experiment_name = str(spec["experiment_name"])
    model_data_file = str(spec["model_data_file"])
    experiment_data = load_experiment_data(
        {"selected": experiment_name},
        str(experiments_dir),
        expected_model_dir=str(config["model_dir"]),
    )["selected"]
    experiment_config = experiment_data["config"]
    _check_checkpoint_model_data(
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        requested_model_data_file=model_data_file,
        allow_mismatch=bool(spec.get("allow_model_data_mismatch", False)),
    )

    exact_cobb_douglas = bool(
        spec.get(
            "exact_cobb_douglas",
            experiment_config.get("exact_cobb_douglas", config.get("exact_cobb_douglas", False)),
        )
    )
    model_data = _load_model_data(
        model_dir / model_data_file,
        object_name=str(spec.get("model_data_object", config["model_data_object"])),
        precision=precision,
    )
    Model = load_model_class(str(config["model_dir"]), exact_cobb_douglas)
    analysis_hooks = load_model_analysis_hooks(str(config["model_dir"]))
    econ_model = Model(
        parameters=model_data["parameters"],
        state_ss=model_data["state_ss"],
        policies_ss=model_data["policies_ss"],
        state_sd=model_data["state_sd"],
        policies_sd=model_data["policies_sd"],
        double_precision=bool(config["double_precision"]),
    )

    nn_config = {
        "features": list(experiment_config["layers"]) + [econ_model.dim_policies],
        "C": model_data["policy_state_matrix"],
        "states_sd": model_data["state_sd"],
        "policies_sd": model_data["policies_sd"],
        "params_dtype": precision,
    }
    train_state = load_trained_model_orbax(
        experiment_name,
        str(experiments_dir),
        nn_config,
        econ_model.state_ss,
        step=spec.get("checkpoint_step"),
    )

    simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config))
    simul_obs, simul_policies, analysis_variables, _ = simulation_analysis(
        train_state=train_state,
        econ_model=econ_model,
        analysis_config=config,
        simulation_fn=simulation_fn,
        analysis_hooks=analysis_hooks,
    )
    mean_consumption, sd_consumption = _consumption_statistics(analysis_variables["Agg. Consumption"])

    welfare_ss = econ_model.utility_ss / (1 - econ_model.beta)
    welfare_fn = jax.jit(get_welfare_fn(econ_model, config))
    welfare_cost = float(
        _compute_welfare_cost_from_sample(
            econ_model=econ_model,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            policies_logdev=simul_policies,
            config_dict=config,
        )
    )

    capital_reallocation = _ergodic_capital_reallocation_pp(
        simul_obs=simul_obs,
        state_ss=econ_model.state_ss,
        policies_ss=econ_model.policies_ss,
        n_sectors=model_data["n_sectors"],
    )
    return {
        "experiment_name": experiment_name,
        "model_data_file": model_data_file,
        "checkpoint_step": spec.get("checkpoint_step", ""),
        "label": spec.get("label", experiment_name),
        "label_tex": spec.get("label_tex"),
        "section": spec.get("section"),
        "section_tex": spec.get("section_tex"),
        "mean_consumption_percent": mean_consumption,
        "sd_consumption_percent": sd_consumption,
        "welfare_cost_percent": welfare_cost,
        "capital_reallocation_pp": capital_reallocation,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_name",
        "model_data_file",
        "checkpoint_step",
        "label",
        "section",
        "mean_consumption_percent",
        "sd_consumption_percent",
        "welfare_cost_percent",
        "capital_reallocation_pp",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}", flush=True)


def _write_latex_table(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{config['caption_tex']}}}",
        rf"\label{{{config['table_label']}}}",
        r"{\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Experiment} & \textbf{Mean Cons.} & \textbf{S.D. Cons.} & "
        r"\textbf{Welfare cost} & \textbf{Capital realloc.} \\",
        r" & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(p.p.)} \\",
        r"\midrule",
    ]
    previous_section: tuple[str | None, str | None] | None = None
    for row in rows:
        section = (row.get("section"), row.get("section_tex"))
        if section != previous_section and any(value is not None for value in section):
            if previous_section is not None:
                lines.append(r"\addlinespace[0.25em]")
            section_text = _display_text(
                row,
                "section",
                "section_tex",
                str(row.get("section") or ""),
            )
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{section_text}}}}} \\")
            previous_section = section

        label = _display_text(row, "label", "label_tex", str(row["experiment_name"]))
        lines.append(
            f"{label} & {row['mean_consumption_percent']:.3f} & "
            f"{row['sd_consumption_percent']:.3f} & "
            f"{row['welfare_cost_percent']:.3f} & "
            f"{row['capital_reallocation_pp']:.3f} \\\\"
        )

    notes = config.get(
        "notes_tex",
        "Mean and standard deviation of consumption are moments of the ergodic distribution and "
        "are reported as percent log differences from the deterministic steady state. "
        "The welfare cost is the consumption-equivalent cost of business cycles, in percent. "
        "Capital reallocation is the sum of the absolute sectoral capital-share changes defined "
        "as in Figure \\ref{fig:cap_share_ergodic}. Each sectoral change is the percent change in its "
        "ergodic-mean capital share relative to its deterministic-steady-state share. "
        "``Homogeneous shocks'' sets sectoral shock volatilities and persistence parameters to common values.",
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{minipage}{0.96\textwidth}",
            r"\vspace{0.5em}",
            r"\footnotesize",
            rf"\textit{{Notes:}} {notes}",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    latex_code = "\n".join(lines)
    path.write_text(latex_code, encoding="utf-8")
    print(f"Saved: {path}", flush=True)
    return latex_code


def run(config: dict[str, Any]) -> list[dict[str, Any]]:
    config = _deep_merge(DEFAULT_CONFIG, config)
    _validate_config(config)

    if bool(config["double_precision"]):
        jax_config.update("jax_enable_x64", True)
    precision = jnp.float64 if config["double_precision"] else jnp.float32

    model_dir = DATA_ROOT / str(config["model_dir"])
    experiments_dir = model_dir / "experiments"
    output_dir = model_dir / "analysis" / "comparisons" / str(config["comparative_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"extensions_summary_{config['comparative_name']}"
    csv_path = output_dir / f"{stem}.csv"
    table_path = output_dir / f"{stem}.tex"
    config_path = output_dir / f"config_{config['comparative_name']}.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {config_path}", flush=True)

    rows: list[dict[str, Any]] = []
    latex_code = ""
    total = len(config["experiments"])
    for index, spec in enumerate(config["experiments"], start=1):
        print(
            f"\n[{index}/{total}] {spec.get('label', spec['experiment_name'])} " f"({spec['model_data_file']})",
            flush=True,
        )
        try:
            rows.append(
                _run_experiment(
                    config=config,
                    spec=spec,
                    model_dir=model_dir,
                    experiments_dir=experiments_dir,
                    precision=precision,
                )
            )
            _write_csv(csv_path, rows)
            latex_code = _write_latex_table(table_path, rows, config)
        finally:
            jax.clear_caches()
            gc.collect()

    print("\n" + "=" * 72)
    print("FINAL LATEX TABLE")
    print("=" * 72)
    print(latex_code)
    print("=" * 72, flush=True)
    return rows


def main() -> None:
    if IN_COLAB:
        run(config)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Optional path to a multi-experiment JSON config; otherwise use the inline config.",
    )
    args = parser.parse_args()
    if not args.config:
        run(config)
        return

    with open(args.config, encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    overrides = raw_config.get("multi_experiment_analysis", raw_config)
    if not isinstance(overrides, dict):
        raise ValueError("The multi_experiment_analysis section must be a JSON object.")
    run(overrides)


if __name__ == "__main__":
    main()
