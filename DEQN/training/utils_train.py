import os
from copy import deepcopy

from DEQN.analysis.model_hooks import get_states_to_shock
from DEQN.training.plots import (
    plot_learning_rate_schedule,
    plot_training_metrics,
    plot_training_summary,
)


def _resolve_model_data_file(model_dir, configured_name, fallback_names):
    candidate_names = []
    for name in [configured_name, *fallback_names]:
        if name is None or name in candidate_names:
            continue
        candidate_names.append(name)

    for filename in candidate_names:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            if configured_name is not None and filename != configured_name:
                print(f"  Using fallback ModelData file: {filename} (configured '{configured_name}' not found)")
            return filename, path

    raise FileNotFoundError(f"ModelData file not found in {model_dir}. Tried: {candidate_names}")


def _set_derived_training_config(config_dict, *, rollout_multiplier=1):
    config_dict["batch_size"] = config_dict.get("batch_size", 16)
    config_dict["periods_per_step"] = (
        config_dict["periods_per_epis"] * config_dict["epis_per_step"] * rollout_multiplier
    )
    if config_dict["periods_per_step"] % config_dict["batch_size"] != 0:
        raise ValueError(
            "periods_per_step must be divisible by batch_size. "
            f"Got periods_per_step={config_dict['periods_per_step']}, batch_size={config_dict['batch_size']}."
        )
    config_dict["n_batches"] = config_dict["periods_per_step"] // config_dict["batch_size"]
    return config_dict


def _is_ir_finetune_enabled(config_dict):
    return bool((config_dict.get("config_ir_finetune") or {}).get("enabled", False))


def _get_ir_finetune_source_exper_name(config_dict):
    return (config_dict.get("config_ir_finetune") or {}).get("source_exper_name")


def _build_ir_finetune_config(base_config, econ_model, analysis_hooks):
    finetune_options = base_config.get("config_ir_finetune") or {}
    ir_config = deepcopy(base_config)

    for key in [
        "learning_rate",
        "periods_per_epis",
        "epis_per_step",
        "steps_per_epoch",
        "n_epochs",
        "checkpoint_every_n_epochs",
        "mc_draws",
        "init_range",
        "simul_vol_scale",
        "config_eval",
    ]:
        if key in finetune_options and finetune_options[key] is not None:
            ir_config[key] = deepcopy(finetune_options[key])

    suffix = finetune_options.get("exper_suffix", "_IR")
    source_exper_name = finetune_options.get("source_exper_name")
    ir_config["exper_name"] = finetune_options.get("exper_name") or f"{base_config['exper_name']}{suffix}"
    ir_config["restore"] = False
    ir_config["restore_exper_name"] = None
    ir_config["restore_step"] = False
    ir_config.pop("restore_checkpoint_step", None)
    ir_config.pop("resume_target_n_epochs", None)
    ir_config.pop("resume_completed_epochs", None)
    ir_config.pop("resume_original_n_epochs", None)
    if source_exper_name:
        ir_config["ir_finetune_source_exper_name"] = source_exper_name
    if finetune_options.get("source_step") is not None:
        ir_config["ir_finetune_source_step"] = finetune_options["source_step"]
    ir_config["eval_ir_rollouts"] = finetune_options.get("eval_ir_rollouts", True)
    ir_config["record_initial_eval"] = finetune_options.get("record_initial_eval", True)
    ir_config["comment"] = finetune_options.get(
        "comment",
        f"IR fine-tuning initialized from {source_exper_name or base_config['exper_name']}",
    )
    ir_config["ir_finetune_min_shock_size"] = finetune_options.get("min_shock_size", 5.0)
    ir_config["ir_finetune_max_shock_size"] = finetune_options.get("max_shock_size", 25.0)

    if finetune_options.get("states_to_shock") is not None:
        ir_config["states_to_shock"] = list(finetune_options["states_to_shock"])
    if finetune_options.get("ir_sectors_to_plot") is not None:
        ir_config["ir_sectors_to_plot"] = list(finetune_options["ir_sectors_to_plot"])

    ir_config["states_to_shock"] = get_states_to_shock(
        config=ir_config,
        econ_model=econ_model,
        analysis_hooks=analysis_hooks,
    )

    return _set_derived_training_config(ir_config, rollout_multiplier=2)


def _plot_result(result, save_dir, experiment_name):
    metrics = result.get("metrics", {})
    if not metrics.get("checkpointed_steps"):
        print(f"No checkpointed metrics for {experiment_name}; skipping plots.", flush=True)
        return
    plot_training_metrics(training_results=result, save_dir=save_dir, experiment_name=experiment_name, display_dpi=100)
    plot_learning_rate_schedule(
        training_results=result, save_dir=save_dir, experiment_name=experiment_name, display_dpi=100
    )
    plot_training_summary(training_results=result, save_dir=save_dir, experiment_name=experiment_name, display_dpi=100)


def _print_metrics_summary(label, metrics):
    if metrics.get("min_loss") is None:
        print(f"{label} metrics unavailable | Time: {metrics['time_fullexp_minutes']:.1f}m")
        return
    print(
        f"{label} normal Loss: {metrics['min_loss']:.7f} | "
        f"Acc: {metrics['max_mean_acc']:.4f} | "
        f"Time: {metrics['time_fullexp_minutes']:.1f}m"
    )
    if metrics.get("ir_eval_min_loss") is not None:
        print(
            f"{label} IR Loss: {metrics['ir_eval_min_loss']:.7f} | "
            f"Acc: {metrics['ir_eval_max_mean_acc']:.4f}"
        )
