"""Smoke test of APG, DEQN, and time iteration on the shared Stone-Geary RBC."""

from econ_models import StoneGearyRbc

from trainers.train import train


def main():
    model = StoneGearyRbc()
    for algorithm in ("apg", "deqn", "time_iteration"):
        result = train(model, algorithm)
        metrics = ", ".join(f"{name}={value:.6g}" for name, value in result.items())
        print(f"{algorithm}: {metrics}")


if __name__ == "__main__":
    main()
