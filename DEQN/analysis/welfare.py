import jax
import jax.numpy as jnp
from jax import lax


def get_welfare_fn(econ_model, config_analysis):

    def discounted_utility(utilities, beta, terminal_value):
        sequence_length = utilities.shape[0]
        discount_factors = beta ** jnp.arange(sequence_length)
        discounted_utilities = jnp.sum(utilities * discount_factors)
        terminal_contribution = terminal_value * beta**sequence_length
        present_value = discounted_utilities + terminal_contribution
        return present_value

    def welfare_fn(utilities_simul, welfare_ss, rng_key):
        # Generate random start indices for sampling
        start_indices = jax.random.randint(
            rng_key,
            (config_analysis["welfare_n_trajects"],),
            0,
            utilities_simul.shape[0] - config_analysis["welfare_traject_length"] + 1,
        )

        # Use lax.dynamic_slice to extract trajectories in a vectorized way
        def extract_trajectory(start_idx):
            return lax.dynamic_slice(utilities_simul, (start_idx,), (config_analysis["welfare_traject_length"],))

        # Extract all trajectories using vmap
        utility_trajects = jax.vmap(extract_trajectory)(start_indices)

        # Calculate discounted welfare for each trajectory
        welfares = jax.vmap(lambda utilities: discounted_utility(utilities, econ_model.beta, welfare_ss))(
            utility_trajects
        )

        # Average across trajectories
        mean_welfare = jnp.mean(welfares)

        return mean_welfare

    return welfare_fn
