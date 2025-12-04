#!/usr/bin/env python3
"""
Test that an economic model is correctly specified.

A correctly specified model should satisfy:
1. The steady state is a fixed point: step(state_ss, policy_ss, zero_shock) == state_ss
2. The loss function is zero at steady state
"""

import jax.numpy as jnp


def test_model_specification(
    econ_model,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """
    Test that an economic model is correctly specified.

    Args:
        econ_model: Economic model instance with step, loss, and expect_realization methods
        rtol: Relative tolerance for numerical comparisons
        atol: Absolute tolerance for numerical comparisons
        verbose: Whether to print detailed results

    Returns:
        dict: Test results with pass/fail status and detailed metrics
    """
    results = {
        "steady_state_fixed_point": {"passed": False, "details": {}},
        "steady_state_loss_zero": {"passed": False, "details": {}},
        "overall_passed": False,
    }

    # Get steady state in normalized form (should be zeros)
    # The model stores state_ss and policies_ss in log levels
    # Normalized form: (x - x_ss) / x_sd, so at steady state this is 0
    n_states = econ_model.dim_states
    n_policies = econ_model.dim_policies
    n_sectors = econ_model.n_sectors

    # Steady state in normalized coordinates (zeros)
    state_ss_norm = jnp.zeros(n_states)
    policy_ss_norm = jnp.zeros(n_policies)
    zero_shock = jnp.zeros(n_sectors)

    # =========================================================================
    # Test 1: Steady state is a fixed point of the step function
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Test 1: Steady State Fixed Point")
        print("=" * 60)

    state_next = econ_model.step(state_ss_norm, policy_ss_norm, zero_shock)
    state_diff = state_next - state_ss_norm

    max_abs_diff = float(jnp.max(jnp.abs(state_diff)))
    mean_abs_diff = float(jnp.mean(jnp.abs(state_diff)))

    # Check if state returns to itself
    fixed_point_passed = jnp.allclose(state_next, state_ss_norm, rtol=rtol, atol=atol)

    results["steady_state_fixed_point"]["passed"] = bool(fixed_point_passed)
    results["steady_state_fixed_point"]["details"] = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "state_diff": state_diff.tolist(),
    }

    if verbose:
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Result: {'PASSED' if fixed_point_passed else 'FAILED'}")

        if not fixed_point_passed:
            # Show which components have largest deviations
            abs_diff = jnp.abs(state_diff)
            worst_indices = jnp.argsort(abs_diff)[-5:][::-1]
            print("\n  Largest deviations:")
            for idx in worst_indices:
                idx_int = int(idx)
                if idx_int < n_sectors:
                    var_name = f"K[{idx_int}]"
                else:
                    var_name = f"a[{idx_int - n_sectors}]"
                print(f"    {var_name}: {float(state_diff[idx_int]):.2e}")

    # =========================================================================
    # Test 2: Loss function is zero at steady state
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Test 2: Steady State Loss = 0")
        print("=" * 60)

    # Compute expectation at steady state
    # At steady state, next period state and policy are the same as current
    expect_ss = econ_model.expect_realization(state_ss_norm, policy_ss_norm)

    # Compute loss at steady state
    mean_loss, mean_accuracy, min_accuracy, mean_acc_focs, min_acc_focs = econ_model.loss(
        state_ss_norm, expect_ss, policy_ss_norm
    )

    # Check if loss is zero (accuracy should be 1)
    loss_zero_passed = float(mean_loss) < atol and float(mean_accuracy) > (1 - rtol)

    results["steady_state_loss_zero"]["passed"] = bool(loss_zero_passed)
    results["steady_state_loss_zero"]["details"] = {
        "mean_loss": float(mean_loss),
        "mean_accuracy": float(mean_accuracy),
        "min_accuracy": float(min_accuracy),
        "mean_accuracies_by_foc": mean_acc_focs.tolist(),
        "min_accuracies_by_foc": min_acc_focs.tolist(),
    }

    if verbose:
        print(f"  Mean squared loss: {float(mean_loss):.2e}")
        print(f"  Mean accuracy: {float(mean_accuracy):.6f}")
        print(f"  Min accuracy: {float(min_accuracy):.6f}")
        print(f"  Result: {'PASSED' if loss_zero_passed else 'FAILED'}")

        # FOC labels (based on the model structure)
        foc_labels = [
            "C (consumption)",
            "L (labor)",
            "K (capital)",
            "Pm (intermediate price)",
            "M (intermediates)",
            "Mout (intermediate output)",
            "Pk (capital price)",
            "Iout (investment output)",
            "Qrc (resource constraint)",
            "Qdef (Q definition)",
            "Ydef (Y definition)",
            "Cagg_def",
            "Lagg_def",
            "Yagg_def",
            "Iagg_def",
            "Magg_def",
        ]

        print("\n  Accuracy by FOC:")
        for i, (label, mean_acc, min_acc) in enumerate(zip(foc_labels, mean_acc_focs, min_acc_focs)):
            status = "✓" if float(min_acc) > (1 - rtol) else "✗"
            print(f"    {status} {label}: mean={float(mean_acc):.6f}, min={float(min_acc):.6f}")

    # =========================================================================
    # Overall result
    # =========================================================================
    results["overall_passed"] = results["steady_state_fixed_point"]["passed"] and results["steady_state_loss_zero"]["passed"]

    if verbose:
        print("\n" + "=" * 60)
        print(f"OVERALL RESULT: {'ALL TESTS PASSED' if results['overall_passed'] else 'SOME TESTS FAILED'}")
        print("=" * 60)

    return results


if __name__ == "__main__":
    # Example usage - can be run standalone
    print("Run this module with a specific model to test.")
    print("Example: from DEQN.tests.test_model_specification import test_model_specification")

