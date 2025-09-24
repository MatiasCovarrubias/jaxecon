import jax
import jax.numpy as jnp


def get_welfare_fn(econ_model, config_analysis):

    def discounted_utility(utilities, beta, terminal_value):
        discount_factors = jnp.array([beta**t for t in range(len(utilities))])
        present_value = jnp.sum(utilities * discount_factors) + terminal_value * beta ** len(utilities)
        return present_value

    def welfare_fn(utilities_simul, welfare_ss):
        # Generate random start indices for sampling
        start_indices = jax.random.randint(
            jax.random.PRNGKey(0),
            (config_analysis["welfare_n_trajects"],),
            0,
            utilities_simul.shape[0] - config_analysis["welfare_traject_length"] + 1,
        )

        # Sample utility trajectories
        utility_trajects = jnp.array(
            [
                utilities_simul[start_index : start_index + config_analysis["welfare_traject_length"]]
                for start_index in start_indices
            ]
        )

        # Calculate discounted welfare for each trajectory
        welfares = jax.vmap(lambda utilities: discounted_utility(utilities, econ_model.beta, welfare_ss))(
            utility_trajects
        )

        # Average across trajectories
        mean_welfare = jnp.mean(welfares)

        return mean_welfare

    return welfare_fn
