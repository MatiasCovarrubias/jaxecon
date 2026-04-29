from dataclasses import dataclass
from typing import Any

import jax
from jax import random

from DEQN.analysis.model_hooks import compute_analysis_variables
from DEQN.analysis.simul_analysis import (
    compute_analysis_dataset_with_context,
    create_episode_simulation_fn_verbose,
    create_shock_path_simulation_fn,
    simulation_analysis,
    simulation_analysis_with_shocks,
)
from DEQN.analysis.welfare_outputs import (
    WELFARE_BOTH_RECENTERED_LABEL,
    WELFARE_L_FIXED_AT_DSS_LABEL,
    _compute_counterfactual_welfare_cost_from_sample,
    _compute_welfare_cost_from_sample,
)
from DEQN.training.checkpoints import load_trained_model_orbax


@dataclass
class SingleExperimentResult:
    label: str
    raw_simulation_data: dict[str, Any]
    analysis_variables: dict[str, Any]
    welfare_costs: dict[str, Any]
    stochastic_ss_states: dict[str, Any]
    stochastic_ss_policies: dict[str, Any]
    stochastic_ss_data: dict[str, Any]
    stochastic_ss_loss: dict[str, Any]
    gir_data: Any
    nonlinear_method_labels: list[str]

    @classmethod
    def from_mapping(cls, label: str, data: dict[str, Any]) -> "SingleExperimentResult":
        return cls(label=label, **data)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "raw_simulation_data": self.raw_simulation_data,
            "analysis_variables": self.analysis_variables,
            "welfare_costs": self.welfare_costs,
            "stochastic_ss_states": self.stochastic_ss_states,
            "stochastic_ss_policies": self.stochastic_ss_policies,
            "stochastic_ss_data": self.stochastic_ss_data,
            "stochastic_ss_loss": self.stochastic_ss_loss,
            "gir_data": self.gir_data,
            "nonlinear_method_labels": self.nonlinear_method_labels,
        }


def _create_nonlinear_simulation_runners(
    *,
    econ_model,
    config_dict,
    analysis_hooks,
    matlab_common_shock_schedule,
):
    use_long_simulation = bool(config_dict.get("long_simulation", False))
    use_ergodic_price_aggregation = bool(config_dict.get("ergodic_price_aggregation", False))
    ergodic_simulation_fn = jax.jit(create_episode_simulation_fn_verbose(econ_model, config_dict))

    def run_ergodic_simulation(train_state):
        simul_obs, simul_policies, simul_analysis_variables, analysis_context = simulation_analysis(
            train_state=train_state,
            econ_model=econ_model,
            analysis_config=config_dict,
            simulation_fn=ergodic_simulation_fn,
            analysis_hooks=analysis_hooks,
        )
        return {
            "simul_obs": simul_obs,
            "simul_policies": simul_policies,
            "simul_analysis_variables": simul_analysis_variables,
            "analysis_context": analysis_context,
            "simul_obs_full": simul_obs,
            "simul_policies_full": simul_policies,
            "simulation_kind": "ergodic",
        }

    if use_long_simulation:
        print("  Nonlinear simulation mode: long ergodic simulation", flush=True)
        return {
            "primary": run_ergodic_simulation,
        }

    if matlab_common_shock_schedule is None:
        raise ValueError(
            "long_simulation=False requires a MATLAB common-shock schedule in the simulation data file."
        )

    shock_path_simulation_fn = jax.jit(create_shock_path_simulation_fn(econ_model))

    def run_common_shock_simulation(train_state):
        (
            simul_obs,
            simul_policies,
            simul_analysis_variables,
            analysis_context,
            simul_obs_full,
            simul_policies_full,
        ) = simulation_analysis_with_shocks(
            train_state=train_state,
            econ_model=econ_model,
            shock_path=matlab_common_shock_schedule["full_shocks"],
            simulation_fn=shock_path_simulation_fn,
            active_start=matlab_common_shock_schedule["active_start"],
            active_end=matlab_common_shock_schedule["active_end"],
            analysis_config=config_dict,
            analysis_hooks=analysis_hooks,
            label=(f"Common-shock nonlinear simulation ({matlab_common_shock_schedule['reference_method']})"),
        )
        return {
            "simul_obs": simul_obs,
            "simul_policies": simul_policies,
            "simul_analysis_variables": simul_analysis_variables,
            "analysis_context": analysis_context,
            "simul_obs_full": simul_obs_full,
            "simul_policies_full": simul_policies_full,
            "simulation_kind": "common_shock",
        }

    print(
        "  Nonlinear simulation mode: shared MATLAB shock path "
        f"({matlab_common_shock_schedule['reference_method']})",
        flush=True,
    )
    runners = {
        "primary": run_common_shock_simulation,
    }
    if use_ergodic_price_aggregation:
        print(
            "  Fixed-price aggregation: also running an auxiliary long ergodic reference sample.",
            flush=True,
        )
        runners["aggregation_reference"] = run_ergodic_simulation
    return runners


def _compute_stochastic_ss_from_sample(
    *,
    sample_label,
    simul_obs,
    train_state,
    stoch_ss_fn,
    stoch_ss_loss_fn,
    analysis_context,
    econ_model,
    analysis_hooks,
    config_dict,
    required,
):
    try:
        stoch_ss_policy, stoch_ss_obs, stoch_ss_obs_std = stoch_ss_fn(simul_obs, train_state)
        if stoch_ss_obs_std.max() > 0.001:
            raise ValueError("Stochastic steady state standard deviation too large")

        stoch_ss_analysis_variables = compute_analysis_variables(
            econ_model=econ_model,
            state_logdev=stoch_ss_obs,
            policy_logdev=stoch_ss_policy,
            analysis_context=analysis_context,
            analysis_hooks=analysis_hooks,
        )

        loss_results = stoch_ss_loss_fn(
            stoch_ss_obs,
            stoch_ss_policy,
            train_state,
            random.PRNGKey(config_dict["seed"]),
        )
        print(
            f"    {sample_label}: equilibrium accuracy {loss_results['mean_accuracy']:.4f} "
            f"(min: {loss_results['min_accuracy']:.4f})",
            flush=True,
        )
        return {
            "stochastic_ss_state": stoch_ss_obs,
            "stochastic_ss_policy": stoch_ss_policy,
            "stochastic_ss_data": stoch_ss_analysis_variables,
            "stochastic_ss_loss": loss_results,
        }
    except Exception:
        if required:
            raise
        print(f"    Warning: stochastic steady state failed for {sample_label}; skipping this variant.", flush=True)
        return None


def _run_experiment_analysis(
    *,
    experiment_label,
    exp_data,
    save_dir,
    nn_config_base,
    econ_model,
    nonlinear_simulation_runners,
    welfare_fn,
    welfare_ss,
    stoch_ss_fn,
    stoch_ss_loss_fn,
    gir_fn,
    config_dict,
    analysis_hooks,
):
    experiment_config = exp_data["config"]
    experiment_name = exp_data["results"]["exper_name"]

    nn_config = nn_config_base.copy()
    nn_config["features"] = experiment_config["layers"] + [econ_model.dim_policies]

    train_state = load_trained_model_orbax(experiment_name, save_dir, nn_config, econ_model.state_ss)

    selected_results = nonlinear_simulation_runners["primary"](train_state)
    selected_analysis_context = selected_results["analysis_context"]
    aggregation_reference_results = None
    if "aggregation_reference" in nonlinear_simulation_runners:
        aggregation_reference_results = nonlinear_simulation_runners["aggregation_reference"](train_state)
        selected_analysis_variables, selected_analysis_context = compute_analysis_dataset_with_context(
            econ_model=econ_model,
            simul_obs=selected_results["simul_obs"],
            simul_policies=selected_results["simul_policies"],
            analysis_config=config_dict,
            analysis_context=aggregation_reference_results["analysis_context"],
            analysis_hooks=analysis_hooks,
        )
        selected_results = dict(selected_results)
        selected_results["simul_analysis_variables"] = selected_analysis_variables
        selected_results["analysis_context"] = selected_analysis_context
        print(
            f"    {experiment_label}: re-aggregated the primary sample under the auxiliary ergodic reference.",
            flush=True,
        )

    welfare_costs = {}
    raw_simulation_data = {}
    analysis_variables = {}
    stochastic_ss_states = {}
    stochastic_ss_policies = {}
    stochastic_ss_data = {}
    stochastic_ss_loss = {}
    method_labels = []

    def store_variant(
        label,
        sample_results,
        *,
        analysis_context_for_reporting,
        required_stochss,
        stochss_source_results=None,
        stochss_analysis_context=None,
    ):
        welfare_cost_ce = _compute_welfare_cost_from_sample(
            econ_model=econ_model,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            policies_logdev=sample_results["simul_policies"],
            config_dict=config_dict,
        )
        welfare_costs[label] = welfare_cost_ce
        print(f"    {label}: welfare cost (CE) {welfare_cost_ce:.4f}%")

        raw_entry = dict(sample_results)
        raw_entry["analysis_context"] = analysis_context_for_reporting
        raw_simulation_data[label] = raw_entry
        analysis_variables[label] = sample_results["simul_analysis_variables"]
        method_labels.append(label)

        if stochss_source_results is None:
            stochss_source_results = sample_results
        if stochss_analysis_context is None:
            stochss_analysis_context = analysis_context_for_reporting
        stochss_results = _compute_stochastic_ss_from_sample(
            sample_label=label,
            simul_obs=stochss_source_results["simul_obs"],
            train_state=train_state,
            stoch_ss_fn=stoch_ss_fn,
            stoch_ss_loss_fn=stoch_ss_loss_fn,
            analysis_context=stochss_analysis_context,
            econ_model=econ_model,
            analysis_hooks=analysis_hooks,
            config_dict=config_dict,
            required=required_stochss,
        )
        if stochss_results is not None:
            stochastic_ss_states[label] = stochss_results["stochastic_ss_state"]
            stochastic_ss_policies[label] = stochss_results["stochastic_ss_policy"]
            stochastic_ss_data[label] = stochss_results["stochastic_ss_data"]
            stochastic_ss_loss[label] = stochss_results["stochastic_ss_loss"]
        return welfare_cost_ce

    def store_counterfactual_welfare(
        label,
        *,
        recenter_consumption=False,
        recenter_labor=False,
        fix_labor_at_dss=False,
    ):
        try:
            welfare_cost_ce = _compute_counterfactual_welfare_cost_from_sample(
                econ_model=econ_model,
                welfare_fn=welfare_fn,
                welfare_ss=welfare_ss,
                policies_logdev=selected_results["simul_policies"],
                config_dict=config_dict,
                recenter_consumption=recenter_consumption,
                recenter_labor=recenter_labor,
                fix_labor_at_dss=fix_labor_at_dss,
            )
        except ValueError as exc:
            print(f"    Warning: welfare counterfactual {label} skipped ({exc}).", flush=True)
            return None

        welfare_costs[label] = welfare_cost_ce
        print(f"    {label}: welfare cost (CE) {welfare_cost_ce:.4f}%")
        return welfare_cost_ce

    if aggregation_reference_results is not None:
        aggregation_reference_label = f"{experiment_label} (ergodic)"
        aggregation_reference_entry = dict(aggregation_reference_results)
        aggregation_reference_entry["analysis_context"] = aggregation_reference_results["analysis_context"]
        raw_simulation_data[aggregation_reference_label] = aggregation_reference_entry

    stochss_and_gir_source_results = aggregation_reference_results or selected_results
    store_variant(
        experiment_label,
        selected_results,
        analysis_context_for_reporting=selected_analysis_context,
        required_stochss=True,
        stochss_source_results=stochss_and_gir_source_results,
        stochss_analysis_context=stochss_and_gir_source_results["analysis_context"],
    )
    if analysis_hooks is not None and hasattr(analysis_hooks, "compute_welfare_outputs"):
        extra_welfare_costs = analysis_hooks.compute_welfare_outputs(
            experiment_label=experiment_label,
            selected_results=selected_results,
            econ_model=econ_model,
            welfare_fn=welfare_fn,
            welfare_ss=welfare_ss,
            config=config_dict,
        )
        for label, welfare_cost_ce in (extra_welfare_costs or {}).items():
            welfare_costs[label] = welfare_cost_ce
            print(f"    {label}: welfare cost (CE) {welfare_cost_ce:.4f}%")
    else:
        store_counterfactual_welfare(
            f"{experiment_label} ({WELFARE_BOTH_RECENTERED_LABEL})",
            recenter_consumption=True,
            recenter_labor=True,
        )
        store_counterfactual_welfare(
            f"{experiment_label} ({WELFARE_L_FIXED_AT_DSS_LABEL})",
            fix_labor_at_dss=True,
        )

    gir_results = gir_fn(
        stochss_and_gir_source_results["simul_obs"],
        train_state,
        stochss_and_gir_source_results["simul_policies"],
        stochastic_ss_states[experiment_label],
    )

    return {
        "raw_simulation_data": raw_simulation_data,
        "analysis_variables": analysis_variables,
        "welfare_costs": welfare_costs,
        "stochastic_ss_states": stochastic_ss_states,
        "stochastic_ss_policies": stochastic_ss_policies,
        "stochastic_ss_data": stochastic_ss_data,
        "stochastic_ss_loss": stochastic_ss_loss,
        "gir_data": gir_results,
        "nonlinear_method_labels": method_labels,
    }
