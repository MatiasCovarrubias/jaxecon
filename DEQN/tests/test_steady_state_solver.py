"""Test the steady state solver with an RBC model."""

import jax.numpy as jnp
from jax import config as jax_config

from DEQN.algorithm import solve_steady_state
from DEQN.econ_models.rbc import Rbc_capadj

jax_config.update("jax_enable_x64", True)


def test_find_steady_state():
    """
    Demonstrate the workflow: find steady state from an unnormalized model.
    
    When state_ss=0 and policies_ss=0, the solver finds the actual log-levels.
    """
    print("=" * 60)
    print("Finding Steady State (Unnormalized Model)")
    print("=" * 60)

    # Create unnormalized model (state_ss=0, policies_ss=0)
    # Variables are raw log-levels: log(K), log(A), log(I)
    model = Rbc_capadj(
        policies_ss=[0.0],
        precision=jnp.float64,
        beta=0.96,
        alpha=0.3,
        delta=0.1,
    )
    model.state_ss = jnp.array([0.0, 0.0])

    # Initial guess in log-levels
    initial_state = jnp.array([1.0, 0.0])   # log(K)~1 → K~2.7, log(A)=0 → A=1
    initial_policy = jnp.array([-1.0])       # log(I)~-1 → I~0.37

    # Solve
    solution = solve_steady_state(model, initial_state, initial_policy)

    # Results
    K_ss = jnp.exp(solution.state[0])
    A_ss = jnp.exp(solution.state[1])
    I_ss = jnp.exp(solution.policy[0])

    print(f"\nSteady state found:")
    print(f"  K_ss = {K_ss:.4f}")
    print(f"  A_ss = {A_ss:.4f}")
    print(f"  I_ss = {I_ss:.4f}")
    print(f"\nConverged: {solution.converged}, Loss: {solution.loss_value:.2e}")

    # Verify against analytical solution
    beta, alpha, delta = 0.96, 0.3, 0.1
    K_analytical = (alpha / (1 / beta - 1 + delta)) ** (1 / (1 - alpha))
    I_analytical = delta * K_analytical

    print(f"\nAnalytical solution:")
    print(f"  K_ss = {K_analytical:.4f}")
    print(f"  I_ss = {I_analytical:.4f}")

    assert solution.converged
    assert abs(K_ss - K_analytical) < 1e-4
    assert abs(I_ss - I_analytical) < 1e-4

    print("\n✓ Test passed!")
    return solution


if __name__ == "__main__":
    test_find_steady_state()
