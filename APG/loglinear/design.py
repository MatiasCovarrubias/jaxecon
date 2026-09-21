"""LQ design objects for the DEQN–APG comparison.

Order of operations (do not invert):

1. ``lq_design(env)`` calls ``solve_loglinear`` while the environment still has
   unit scales, so ``A``, ``B``, ``D``, ``C``, ``P``, ``costate``, and ``Sigma``
   are in the linearization coordinates of ``linearize_environment`` (normalized
   log state and logit action deviation, unit ``state_sd`` / ``policies_sd``).
2. ``apply_lq_design`` then writes the Lyapunov scales onto the model and the
   environment. Observation vectors become z-scores of those unit-scale
   deviations.
3. Residual networks keep the unit-scale ``C`` and convert with the scales, as
   in ``PolicyNetLoglinear`` / ``NeuralNet_loglinear``.
4. The APG tail uses ``P`` and ``λ`` transformed into the *current* observation
   coordinates (z-scores after step 2), so ``W(x) = λᵀx + ½ xᵀPx`` with the
   steady-state observation at the origin.

Nothing here is adaptive: no control variate, no output-layer preconditioner.
"""

from typing import NamedTuple

import numpy as np
from scipy import linalg as sla

from .solve import print_loglinear_solution, solve_loglinear


class LQDesign(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    C: np.ndarray
    P: np.ndarray
    costate: np.ndarray
    A_closed: np.ndarray
    Sigma: np.ndarray
    states_sd: np.ndarray
    policies_sd: np.ndarray
    foc_residual: np.ndarray
    spectral_radius: float

    def to_config(self):
        return {
            "lq_A": np.asarray(self.A).tolist(),
            "lq_B": np.asarray(self.B).tolist(),
            "lq_D": np.asarray(self.D).tolist(),
            "lq_C": np.asarray(self.C).tolist(),
            "lq_P": np.asarray(self.P).tolist(),
            "lq_costate": np.asarray(self.costate).tolist(),
            "lq_A_closed": np.asarray(self.A_closed).tolist(),
            "lq_Sigma": np.asarray(self.Sigma).tolist(),
            "lq_states_sd": np.asarray(self.states_sd).tolist(),
            "lq_policies_sd": np.asarray(self.policies_sd).tolist(),
            "lq_foc_residual": np.asarray(self.foc_residual).tolist(),
            "lq_rho_closed": float(self.spectral_radius),
            "loglinear_C": np.asarray(self.C).tolist(),
            "loglinear_states_sd": np.asarray(self.states_sd).tolist(),
            "loglinear_policies_sd": np.asarray(self.policies_sd).tolist(),
            "loglinear_spectral_radius": float(self.spectral_radius),
        }


def design_from_config(config):
    C = np.asarray(config["lq_C"] if "lq_C" in config else config["loglinear_C"], dtype=np.float64)
    A = np.asarray(config["lq_A"], dtype=np.float64)
    B = np.asarray(config["lq_B"], dtype=np.float64)
    return LQDesign(
        A=A,
        B=B,
        D=np.asarray(config["lq_D"], dtype=np.float64),
        C=C,
        P=np.asarray(config["lq_P"], dtype=np.float64),
        costate=np.asarray(config["lq_costate"], dtype=np.float64),
        A_closed=np.asarray(config.get("lq_A_closed", A + B @ C), dtype=np.float64),
        Sigma=np.asarray(config["lq_Sigma"], dtype=np.float64),
        states_sd=np.asarray(config["lq_states_sd"], dtype=np.float64),
        policies_sd=np.asarray(config["lq_policies_sd"], dtype=np.float64),
        foc_residual=np.asarray(config["lq_foc_residual"], dtype=np.float64),
        spectral_radius=float(config["lq_rho_closed"]),
    )


def lq_design(env, solver="scipy"):
    """Solve the unit-scale LQ policy and the closed-loop stationary covariance."""
    solution = solve_loglinear(env, solver=solver)
    A_closed = np.asarray(solution.A_closed, dtype=np.float64)
    D = np.asarray(solution.D, dtype=np.float64)
    Sigma = sla.solve_discrete_lyapunov(A_closed, D @ D.T)
    Sigma = 0.5 * (Sigma + Sigma.T)
    design = LQDesign(
        A=np.asarray(solution.A, dtype=np.float64),
        B=np.asarray(solution.B, dtype=np.float64),
        D=D,
        C=np.asarray(solution.C, dtype=np.float64),
        P=np.asarray(solution.P, dtype=np.float64),
        costate=np.asarray(solution.costate, dtype=np.float64),
        A_closed=A_closed,
        Sigma=Sigma,
        states_sd=np.asarray(solution.states_sd, dtype=np.float64),
        policies_sd=np.asarray(solution.policies_sd, dtype=np.float64),
        foc_residual=np.asarray(solution.foc_residual, dtype=np.float64),
        spectral_radius=float(solution.spectral_radius),
    )
    print_loglinear_solution(solution)
    print(f"  FOC residual (LQ design): {design.foc_residual}")
    print(f"  rho(A+BC): {design.spectral_radius:.6f}")
    return design


def _scale_quadratic(P, costate, states_sd):
    """Map a unit-scale quadratic into current (z-score) observation coordinates."""
    scale = np.asarray(states_sd, dtype=np.float64).reshape(-1)
    P_obs = P * np.outer(scale, scale)
    costate_obs = costate * scale
    return P_obs, costate_obs


def _obs_covariance(Sigma_unit, states_sd):
    scale = np.asarray(states_sd, dtype=np.float64).reshape(-1)
    return Sigma_unit / np.outer(scale, scale)


def knobs_from_config(config, *, default_on=False):
    """Read residual / scales / start / tail flags.

    ``apply_lq_design`` passes ``default_on=True`` so an empty config still
    applies the old all-on bundle. Trainers use the default (off) so a plain
    run does not pick up scales or a stationary start by accident.
    """
    has_explicit = any(
        key in config for key in ("lq_residual", "lq_scales", "lq_stationary_start")
    )
    if has_explicit:
        residual = bool(config.get("lq_residual", config.get("loglinear_baseline", False)))
        scales = bool(config.get("lq_scales", False))
        stationary_start = bool(config.get("lq_stationary_start", False))
    elif default_on or config.get("design") == "lq" or config.get("loglinear_baseline"):
        residual = bool(config.get("loglinear_baseline", True))
        scales = bool(config.get("loglinear_normalize", True))
        stationary_start = True
    else:
        residual = False
        scales = False
        stationary_start = False
    tail = bool(config.get("use_lq_terminal_value", False))
    return {
        "residual": residual,
        "scales": scales,
        "stationary_start": stationary_start,
        "tail": tail,
    }


def needs_lq_objects(config):
    knobs = knobs_from_config(config)
    return any(knobs.values()) or "lq_A" in config or "lq_C" in config


def inject_lq_design(
    config,
    design,
    *,
    residual=True,
    scales=True,
    stationary_start=True,
    apg_tail=False,
):
    """Write LQ matrices and the selected comparison knobs into ``config``."""
    config.update(design.to_config())
    config["lq_residual"] = bool(residual)
    config["lq_scales"] = bool(scales)
    config["lq_stationary_start"] = bool(stationary_start)
    config["loglinear_baseline"] = bool(residual)
    config["loglinear_normalize"] = bool(scales)
    config["antithetic_episodes"] = True
    if any((residual, scales, stationary_start, apg_tail)):
        config["design"] = "lq"
    if apg_tail:
        config["use_lq_terminal_value"] = True
        config["use_model_terminal_value"] = False
        config["use_terminal_value"] = False
    else:
        config["use_lq_terminal_value"] = False
    return config


def apply_lq_design(env, model, config, design=None):
    """Apply the selected LQ knobs in current observation coordinates.

    ``env`` is the APG environment used to solve (or already holding) the design.
    ``model`` is the object whose ``set_scales`` / ``set_stationary_start`` the
    trainers read (the DEQN model, or ``env.econ``). Both are updated when they
    are distinct objects. Scales and the stationary start are independent: the
    tail quadratic is written in whatever observation coordinates are in force.
    """
    if design is None:
        if "lq_C" in config or "lq_A" in config:
            design = design_from_config(config)
        else:
            if env is None:
                raise ValueError("apply_lq_design needs an environment to solve the LQ design")
            design = lq_design(env, solver=config.get("loglinear_solver", "scipy"))

    knobs = knobs_from_config(config, default_on=True)
    targets = []
    for obj in (model, env):
        if obj is not None and obj not in targets:
            targets.append(obj)

    if knobs["scales"]:
        for obj in targets:
            obj.set_scales(design.states_sd, design.policies_sd)
        P_obs, costate_obs = _scale_quadratic(design.P, design.costate, design.states_sd)
        Sigma_obs = _obs_covariance(design.Sigma, design.states_sd)
    else:
        P_obs, costate_obs = np.asarray(design.P), np.asarray(design.costate)
        Sigma_obs = np.asarray(design.Sigma)

    if knobs["stationary_start"]:
        for obj in targets:
            obj.set_stationary_start(Sigma_obs)

    config.update(design.to_config())
    config["lq_terminal_P"] = np.asarray(P_obs).tolist()
    config["lq_terminal_costate"] = np.asarray(costate_obs).tolist()
    config["lq_Sigma_obs"] = np.asarray(Sigma_obs).tolist()
    config["lq_residual"] = knobs["residual"]
    config["lq_scales"] = knobs["scales"]
    config["lq_stationary_start"] = knobs["stationary_start"]
    config["loglinear_baseline"] = knobs["residual"]
    config["loglinear_normalize"] = knobs["scales"]
    eval_cfg = config.get("config_eval")
    if isinstance(eval_cfg, dict):
        eval_cfg["initial_state_mode"] = "box"
    return design
