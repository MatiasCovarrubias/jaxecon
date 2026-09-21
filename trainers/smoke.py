"""Smoke test of APG, DEQN, and time iteration on the shared Stone-Geary RBC."""

from econ_models import StoneGearyRbc

from trainers.evaluate import compare, format_comparison
from trainers.train import train


def main():
    model = StoneGearyRbc()
    neural = {"episodes": 8, "periods": 32, "epochs": 100, "steps_per_epoch": 1}
    runs = (
        ("apg", neural),
        ("deqn", neural),
        ("time_iteration", {"ti_iterations": 100}),
    )
    policies = {}
    for algorithm, config in runs:
        solution = train(model, algorithm, config)
        policies[algorithm] = solution.policy
        metrics = ", ".join(f"{name}={value:.6g}" for name, value in solution.metrics.items())
        print(f"{algorithm}: {metrics}", flush=True)
    print(format_comparison(compare(model, policies)), flush=True)


if __name__ == "__main__":
    main()
