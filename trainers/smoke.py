"""Train APG, DEQN, and time iteration on the shared Stone-Geary RBC.

``python -m trainers.smoke`` uses the locked comparison settings.
``python -m trainers.smoke --short`` is a few steps of the same code.
"""

import sys

from econ_models import StoneGearyRbc

from trainers.evaluate import evaluate, format_metrics
from trainers.policy import split_streams
from trainers.train import experiment_config, train


def main():
    short = "--short" in sys.argv
    config = experiment_config()
    eval_config = {"eval_episodes": 32, "eval_periods": 2048, "eval_mc_draws": 32}
    if short:
        config.update(
            epochs=1,
            steps_per_epoch=1,
            episodes=4,
            periods=8,
            tail_periods=4,
            n_a=5,
            n_k=9,
            ti_iterations=3,
            newton_steps=4,
        )
        eval_config = {"eval_episodes": 4, "eval_periods": 16, "eval_mc_draws": 4}
    seeds = (0,) if short else (0, 1, 2, 3)
    model = StoneGearyRbc()
    time_iteration = train(model, "time_iteration", config)
    print(f"time_iteration: iterations={time_iteration.metrics['iterations']:.0f}", flush=True)
    totals = {"apg": [], "deqn": [], "time_iteration": []}
    for seed in seeds:
        config["seed"] = seed
        policies = {"time_iteration": time_iteration.policy}
        wall_seconds = {"time_iteration": time_iteration.metrics["wall_seconds"]}
        for algorithm in ("apg", "deqn"):
            solution = train(model, algorithm, config)
            policies[algorithm] = solution.policy
            wall_seconds[algorithm] = solution.metrics["wall_seconds"]
        scored = dict(eval_config)
        scored["eval_rng"] = split_streams(seed)["eval"]
        results = evaluate(model, policies, scored)
        for algorithm, seconds in wall_seconds.items():
            results[algorithm]["wall_seconds"] = seconds
            totals[algorithm].append(results[algorithm]["ce_vs_ss"])
        print(f"\nseed {seed}", flush=True)
        print(format_metrics(results), flush=True)
    if len(seeds) > 1:
        print("\nmean ce_vs_ss_%", flush=True)
        for algorithm, values in totals.items():
            print(f"{algorithm:<16} {100 * sum(values) / len(values):12.6g}", flush=True)


if __name__ == "__main__":
    main()
