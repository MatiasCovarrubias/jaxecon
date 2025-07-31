import jax
import jax.numpy as jnp
import pytest


def test_steady_state_invariance(model, rng):
    """Test that steady state is preserved under zero shock."""
    obs = jnp.zeros_like(model.obs_ss)
    pol = jnp.ones_like(model.policies_ss)
    shock = jnp.zeros_like(model.sample_shock(rng))

    next_obs = model.step(obs, pol, shock)
    assert jnp.allclose(next_obs, obs, atol=1e-10), "Steady state not preserved under zero shock"

    # Test loss at steady state
    expect = model.expect_realization(obs, pol)
    mean_loss, mean_accuracy, min_accuracy, *_ = model.loss(obs, expect, pol)
    assert mean_loss < 1e-12, "Non-zero loss at steady state"
    assert mean_accuracy > 0.999, "Low accuracy at steady state"
    assert min_accuracy > 0.999, "Low minimum accuracy at steady state"


@pytest.mark.parametrize("eps", [1e-4, 1e-5])
def test_linear_vs_nonlinear_dynamics(model, rng, eps):
    """Test consistency between linearized and full dynamics for small shocks."""

    shock = model.sample_shock(rng) * eps
    obs = jnp.zeros_like(model.obs_ss)
    pol = jnp.ones_like(model.policies_ss)

    obs_lin = model.step_loglinear(obs, shock)
    obs_nonlin = model.step(obs, pol, shock)

    assert jnp.allclose(
        obs_lin, obs_nonlin, atol=eps**2
    ), f"Linear and nonlinear dynamics differ by more than O(eps^2) for eps={eps}"


def test_shock_statistics(model, rng):
    """Test statistical properties of shock generation."""
    n_samples = 10000

    # Test single shock sampling
    shock = model.sample_shock(rng)
    assert shock.shape == (model.n_sectors,), "Incorrect shock shape"

    # Test Monte Carlo shock statistics
    mc_shocks = model.mc_shocks(rng, mc_draws=n_samples)
    assert mc_shocks.shape == (n_samples, model.n_sectors), "Incorrect MC shock shape"

    # Check mean and covariance
    empirical_mean = jnp.mean(mc_shocks, axis=0)
    empirical_cov = jnp.cov(mc_shocks.T)

    assert jnp.allclose(empirical_mean, 0.0, atol=1e-2), "Shock mean significantly different from zero"
    assert jnp.allclose(empirical_cov, model.Sigma_A, atol=1e-1), "Shock covariance differs from Sigma_A"


def test_batch_consistency(model, rng):
    """Test consistency between vectorized and sequential operations."""
    batch_size = 10
    rng_keys = jax.random.split(rng, batch_size)

    # Generate batch of observations and shocks
    obs_batch = jnp.stack([model.initial_obs(k) for k in rng_keys])
    shock_batch = jnp.stack([model.sample_shock(k) for k in rng_keys])
    pol = jnp.ones_like(model.policies_ss)

    # Compare vectorized vs sequential computation
    next_obs_batch = jax.vmap(model.step, in_axes=(0, None, 0))(obs_batch, pol, shock_batch)
    next_obs_seq = jnp.stack([model.step(obs, pol, shock) for obs, shock in zip(obs_batch, shock_batch)])

    assert jnp.allclose(
        next_obs_batch, next_obs_seq, atol=1e-10
    ), "Vectorized and sequential operations give different results"


def test_jit_compatibility(model, rng):
    """Test that key methods can be JIT-compiled."""
    obs = model.initial_obs(rng)
    pol = jnp.ones_like(model.policies_ss)
    shock = model.sample_shock(rng)

    # Test JIT compilation of key methods
    jitted_step = jax.jit(model.step)
    jitted_loss = jax.jit(lambda o, p: model.loss(o, model.expect_realization(o, p), p))

    try:
        next_obs = jitted_step(obs, pol, shock)
        loss_vals = jitted_loss(obs, pol)
        assert True, "JIT compilation successful"
    except Exception as e:
        pytest.fail(f"JIT compilation failed: {str(e)}")


@pytest.mark.optional
def test_gradient_stability(model, rng):
    """Test that gradients can be computed without numerical issues."""
    obs = model.initial_obs(rng)
    pol = jnp.ones_like(model.policies_ss)

    def loss_fn(policy):
        expect = model.expect_realization(obs, policy)
        return model.loss(obs, expect, policy)[0]

    try:
        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(pol)
        assert not jnp.any(jnp.isnan(grad_val)), "NaN in gradient"
        assert not jnp.any(jnp.isinf(grad_val)), "Inf in gradient"
    except Exception as e:
        pytest.fail(f"Gradient computation failed: {str(e)}")


def test_linear_simulations(model, rng):
    """Test running multiple parallel simulations using linear dynamics."""
    n_simulations = 10
    n_periods = 1000

    # Split RNG for initial conditions and shock sequences
    rng_init, rng_shocks = jax.random.split(rng)

    # Generate initial conditions for all simulations
    rng_keys_init = jax.random.split(rng_init, n_simulations)
    init_obs_batch = jax.vmap(lambda k: model.initial_obs(k, range=1))(rng_keys_init)

    # Generate shock sequences for all simulations and periods
    rng_keys_shocks = jax.random.split(rng_shocks, n_periods)
    shock_batch = jax.vmap(model.sample_shock)(rng_keys_shocks)  # (n_periods, n_sectors)

    # Define single period step function
    def period_step(obs, shock):
        obs_next = model.step_loglinear(obs, shock)
        return obs_next, obs_next

    # Run parallel simulations using vmap and scan
    _, trajectories = jax.lax.scan(
        lambda obs, shock: jax.vmap(period_step, in_axes=(0, None))(obs, shock), init_obs_batch, shock_batch
    )

    # Basic checks on output shape and values
    assert trajectories.shape == (
        n_periods,
        n_simulations,
        model.dim_obs,
    ), f"Expected shape {(n_periods, n_simulations, model.dim_obs)}, got {trajectories.shape}"

    # Check for NaN and Inf values
    assert not jnp.any(jnp.isnan(trajectories)), "NaN values in simulation trajectories"
    assert not jnp.any(jnp.isinf(trajectories)), "Infinite values in simulation trajectories"

    # Check maximum deviation
    max_dev = jnp.max(jnp.abs(trajectories))
    assert max_dev < 10.0, f"Unreasonably large deviations in trajectories: {max_dev}"

    # Compute statistics across all simulations
    flat_trajectories = trajectories.reshape(-1, model.dim_obs)
    means = jnp.mean(flat_trajectories, axis=0)
    stds = jnp.std(flat_trajectories, axis=0)

    # Check if means are close to 0 and stds are close to 1
    assert jnp.allclose(means, 0.0, atol=0.1), f"Means significantly different from 0: {means}"
    assert jnp.allclose(stds, 1.0, atol=0.1), f"Standard deviations significantly different from 1: {stds}"


def run_model_tests(model):
    """Run all tests for the RBC Production Network model.

    Args:
        model: An instantiated RBC Production Network model

    Returns:
        dict: Results of all tests with their status
    """
    # Create a consistent RNG for all tests
    rng = jax.random.PRNGKey(42)
    results = {}

    try:
        test_steady_state_invariance(model, rng)
        results["steady_state"] = "✓ Passed"
    except AssertionError as e:
        results["steady_state"] = f"✗ Failed: {str(e)}"

    try:
        for eps in [1e-4, 1e-5]:
            test_linear_vs_nonlinear_dynamics(model, rng, eps)
        results["linear_vs_nonlinear"] = "✓ Passed"
    except AssertionError as e:
        results["linear_vs_nonlinear"] = f"✗ Failed: {str(e)}"

    try:
        test_shock_statistics(model, rng)
        results["shock_statistics"] = "✓ Passed"
    except AssertionError as e:
        results["shock_statistics"] = f"✗ Failed: {str(e)}"

    try:
        test_batch_consistency(model, rng)
        results["batch_consistency"] = "✓ Passed"
    except AssertionError as e:
        results["batch_consistency"] = f"✗ Failed: {str(e)}"

    try:
        test_jit_compatibility(model, rng)
        results["jit_compatibility"] = "✓ Passed"
    except AssertionError as e:
        results["jit_compatibility"] = f"✗ Failed: {str(e)}"

    try:
        test_gradient_stability(model, rng)
        results["gradient_stability"] = "✓ Passed"
    except AssertionError as e:
        results["gradient_stability"] = f"✗ Failed: {str(e)}"

    try:
        test_linear_simulations(model, rng)
        results["parallel_linear_simulations"] = "✓ Passed"
    except AssertionError as e:
        results["parallel_linear_simulations"] = f"✗ Failed: {str(e)}"

    return results


if __name__ == "__main__":
    pytest.main([__file__])
