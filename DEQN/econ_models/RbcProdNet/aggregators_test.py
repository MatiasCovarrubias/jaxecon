import jax
import jax.numpy as jnp
import pytest
from aggregators import tornqvist_index_vectorized


def generate_ar1_simple(rng_key, T, n_sectors, rho, sigma):
    """Simple AR1 generation using scan."""

    def ar1_step(x_prev, shock):
        x_next = rho * x_prev + shock
        return x_next, x_next

    # Generate shocks and initial state
    rng1, rng2 = jax.random.split(rng_key)
    shocks = jax.random.normal(rng1, (T - 1, n_sectors)) * sigma
    x0 = jax.random.normal(rng2, (n_sectors,)) * sigma / jnp.sqrt(1 - rho**2)

    # Generate AR1 series
    _, x_series = jax.lax.scan(ar1_step, x0, shocks)
    full_series = jnp.concatenate([x0[None, :], x_series], axis=0)

    return jnp.exp(full_series)  # Convert to positive levels


def test_tornqvist_index_starts_at_zero():
    """Test that Tornqvist index always starts at zero."""
    rng = jax.random.PRNGKey(42)
    T, n_sectors = 10, 3

    quantities = jax.random.uniform(rng, (T, n_sectors), minval=0.5, maxval=2.0)
    prices = jax.random.uniform(rng, (T, n_sectors), minval=0.5, maxval=2.0)

    index = tornqvist_index_vectorized(quantities, prices)

    assert jnp.allclose(index[0], 0.0, atol=1e-10), "Tornqvist index should start at zero"
    assert index.shape == (T,), f"Index should have shape ({T},), got {index.shape}"


def test_tornqvist_index_fixed_quantities():
    """Test Tornqvist index with constant quantities."""
    T, n_sectors = 10, 3

    quantities = jnp.ones((T, n_sectors))
    prices = jnp.ones((T, n_sectors)) * 2.0

    index = tornqvist_index_vectorized(quantities, prices)

    # With no quantity growth, index should remain at zero
    expected_index = jnp.zeros(T)
    assert jnp.allclose(index, expected_index, atol=1e-10), "Index should remain zero when quantities are constant"


def test_tornqvist_index_uniform_growth():
    """Test Tornqvist index with uniform growth."""
    T, n_sectors = 5, 3
    growth_rate = 0.02

    base_quantities = jnp.ones(n_sectors)
    quantities = jnp.array([base_quantities * (1 + growth_rate) ** t for t in range(T)])
    prices = jnp.ones((T, n_sectors))

    index = tornqvist_index_vectorized(quantities, prices)

    expected_growth_per_period = jnp.log(1 + growth_rate)
    expected_index = jnp.array([expected_growth_per_period * t for t in range(T)])

    assert jnp.allclose(index, expected_index, atol=1e-6), "Index should equal cumulative log growth"


def test_tornqvist_index_ar1_mean_reversion():
    """Test that Tornqvist index mean approaches zero with stationary AR1 processes."""
    base_rng = jax.random.PRNGKey(42)
    n_sectors = 3
    rho = 0.7  # Stationary
    sigma = 0.1

    # Test that longer series have means closer to zero
    index_means = []

    for T in [100, 500]:
        sim_means = []

        # Run a few simulations
        for i in range(20):
            rng = jax.random.fold_in(base_rng, i * T)
            rng1, rng2 = jax.random.split(rng)

            quantities = generate_ar1_simple(rng1, T, n_sectors, rho, sigma)
            prices = generate_ar1_simple(rng2, T, n_sectors, rho, sigma)

            index = tornqvist_index_vectorized(quantities, prices)
            sim_means.append(jnp.mean(index))

        avg_mean = jnp.mean(jnp.array(sim_means))
        index_means.append(avg_mean)

    # Check that mean is small and gets smaller with longer series
    assert jnp.abs(index_means[0]) < 0.1, f"Mean should be small for T=100, got {index_means[0]:.4f}"
    assert jnp.abs(index_means[1]) < jnp.abs(index_means[0]), "Mean should get smaller with longer series"


def test_tornqvist_index_price_independence():
    """Test that index only depends on quantities, not prices."""
    T, n_sectors = 5, 3

    quantities = jnp.ones((T, n_sectors))

    # Two different price series
    prices1 = jnp.ones((T, n_sectors))
    prices2 = jnp.array([[1.0, 2.0, 3.0] for _ in range(T)]) * jnp.arange(1, T + 1)[:, None]

    index1 = tornqvist_index_vectorized(quantities, prices1)
    index2 = tornqvist_index_vectorized(quantities, prices2)

    # Both should be zero since quantities don't change
    assert jnp.allclose(index1, jnp.zeros(T), atol=1e-10), "Index should be zero with constant quantities"
    assert jnp.allclose(index2, jnp.zeros(T), atol=1e-10), "Index should be zero regardless of price changes"


def run_aggregator_tests():
    """Run all tests for the aggregator functions."""
    results = {}

    test_functions = [
        ("tornqvist_starts_at_zero", test_tornqvist_index_starts_at_zero),
        ("tornqvist_fixed_quantities", test_tornqvist_index_fixed_quantities),
        ("tornqvist_uniform_growth", test_tornqvist_index_uniform_growth),
        ("tornqvist_ar1_mean_reversion", test_tornqvist_index_ar1_mean_reversion),
        ("tornqvist_price_independence", test_tornqvist_index_price_independence),
    ]

    for test_name, test_func in test_functions:
        try:
            test_func()
            results[test_name] = "✓ Passed"
        except AssertionError as e:
            results[test_name] = f"✗ Failed: {str(e)}"
        except Exception as e:
            results[test_name] = f"✗ Error: {str(e)}"

    return results


if __name__ == "__main__":
    results = run_aggregator_tests()

    print("Aggregator Test Results:")
    print("=" * 50)
    for test_name, result in results.items():
        print(f"{test_name:30} {result}")

    passed = sum(1 for r in results.values() if r.startswith("✓"))
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed")
