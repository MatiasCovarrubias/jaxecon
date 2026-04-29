from typing import Any

WELFARE_OUTPUT_ORDER = [
    "Global Solution",
    "C and L recentered at determ SS",
    "L fixed at determ SS",
    "FirstOrder",
    "MITShocks",
    "PerfectForesight",
]


def build_output_display_label_map(config_dict: dict[str, Any]) -> dict[str, str]:
    experiment_config = config_dict.get("experiment_to_analyze") or config_dict.get("experiments_to_analyze") or {}
    experiment_labels = list(experiment_config.keys())
    if len(experiment_labels) != 1:
        return {}

    experiment_label = experiment_labels[0]
    return {
        experiment_label: "Global Solution",
        f"{experiment_label} (ergodic)": "Ergodic Reference",
        f"{experiment_label} (common shocks)": "Global Solution (common shocks)",
        f"{experiment_label} (C and L recentered at determ SS)": "C and L recentered at determ SS",
        f"{experiment_label} (L fixed at determ SS)": "L fixed at determ SS",
    }


def apply_display_labels_to_mapping(values_by_label, label_map):
    if not label_map:
        return dict(values_by_label)

    relabeled = {}
    for label, value in values_by_label.items():
        relabeled[label_map.get(label, label)] = value
    return relabeled


def apply_display_labels_to_postprocess_context(postprocess_context, label_map):
    if not postprocess_context or not label_map:
        return postprocess_context

    relabeled_context = dict(postprocess_context)

    ergodic_labels = postprocess_context.get("ergodic_experiment_labels")
    if ergodic_labels is not None:
        relabeled_context["ergodic_experiment_labels"] = [label_map.get(label, label) for label in ergodic_labels]

    reference_label = postprocess_context.get("reference_experiment_label")
    if reference_label is not None:
        relabeled_context["reference_experiment_label"] = label_map.get(reference_label, reference_label)

    return relabeled_context


def apply_display_labels_to_sequence(labels, label_map):
    if labels is None:
        return None
    return [label_map.get(label, label) for label in labels]


def build_display_welfare_costs(welfare_costs, label_map):
    display_welfare_costs = apply_display_labels_to_mapping(welfare_costs, label_map)
    ordered_welfare_costs = {
        label: display_welfare_costs[label] for label in WELFARE_OUTPUT_ORDER if label in display_welfare_costs
    }
    return ordered_welfare_costs or display_welfare_costs
