from jax import numpy as jnp
from jax import random
import jax


class RbcProdNetAggregators:
    """Aggregation methods for RBC Production Network model."""

    def __init__(self, model):
        """Initialize with reference to base model."""
        self.model = model

    def aggregator_fixed_prices(self, simul_obs, simul_policy, Pvec, Pkvec, Pmvec):
        """
        Creating new aggregator funcion. The objective is to aggregate K, C, M, I, Y using new prices Pvec, Pkvec, Pmvec.

        Input:
        simul_obs has dimension (T_simul, 3*n_sectors) or (3*n_sectors,).
        simul_policy has dimension (T_simul, 11*n_sectors+5) or (11*n_sectors+5,).
        Pvec, Pkvec, and Pmvec have dimension (n_sectors,).

        Process:
        C and Y aggregate with Pvec. K and I aggregate with Pkvec. M aggregates with Pmvec.

        Output:
        Dictionary with new series for Kagg, Cagg, Magg, Iagg, Yagg. For Cagg we call it Cagg_prices.
        """

        # Handle vector inputs by adding time dimension
        is_vector_input = False
        if simul_obs.ndim == 1:
            simul_obs = simul_obs.reshape(1, -1)  # (3*n_sectors,) -> (1, 3*n_sectors)
            is_vector_input = True
        if simul_policy.ndim == 1:
            simul_policy = simul_policy.reshape(1, -1)  # (11*n_sectors+5,) -> (1, 11*n_sectors+5)

        # process obs and policy - broadcasting works element-wise across time and variables
        obs_notnorm = simul_obs * self.model.obs_sd + self.model.obs_ss  # denormalize
        policy_notnorm = simul_policy * jnp.exp(self.model.policies_ss)

        # Extract variables by indexing columns (second dimension)
        K = jnp.exp(obs_notnorm[:, : self.model.n_sectors])  # Shape: (T_simul, n_sectors)
        C = policy_notnorm[:, : self.model.n_sectors]  # Shape: (T_simul, n_sectors)
        M = policy_notnorm[:, 4 * self.model.n_sectors : 5 * self.model.n_sectors]
        I = policy_notnorm[:, 6 * self.model.n_sectors : 7 * self.model.n_sectors]
        Y = policy_notnorm[:, 10 * self.model.n_sectors : 11 * self.model.n_sectors]

        # Aggregate across sectors (axis=1) for each time period
        Kagg = jnp.sum(K * Pkvec, axis=1)  # Sum across sectors, result: (T_simul,)
        Cagg_prices = jnp.sum(C * Pvec, axis=1)
        Magg = jnp.sum(M * Pmvec, axis=1)
        Iagg = jnp.sum(I * Pkvec, axis=1)
        Yagg = jnp.sum(Y * Pvec, axis=1)

        # Get deterministic steady state values
        K_detss = jnp.exp(self.model.obs_ss[: self.model.n_sectors])
        Kagg_detss = jnp.dot(K_detss, Pkvec)
        Kagg_devs = jnp.log(Kagg / Kagg_detss)

        C_detss = jnp.exp(self.model.policies_ss[: self.model.n_sectors])
        Cagg_detss = jnp.dot(C_detss, Pvec)
        Cagg_devs = jnp.log(Cagg_prices / Cagg_detss)

        M_detss = jnp.exp(self.model.policies_ss[4 * self.model.n_sectors : 5 * self.model.n_sectors])
        Magg_detss = jnp.dot(M_detss, Pmvec)
        Magg_devs = jnp.log(Magg / Magg_detss)

        I_detss = jnp.exp(self.model.policies_ss[6 * self.model.n_sectors : 7 * self.model.n_sectors])
        Iagg_detss = jnp.dot(I_detss, Pkvec)
        Iagg_devs = jnp.log(Iagg / Iagg_detss)

        Y_detss = jnp.exp(self.model.policies_ss[10 * self.model.n_sectors : 11 * self.model.n_sectors])
        Yagg_detss = jnp.dot(Y_detss, Pvec)
        Yagg_devs = jnp.log(Yagg / Yagg_detss)

        # If input was vector, squeeze output to remove time dimension
        if is_vector_input:
            Kagg_devs = jnp.squeeze(Kagg_devs)
            Cagg_devs = jnp.squeeze(Cagg_devs)
            Magg_devs = jnp.squeeze(Magg_devs)
            Iagg_devs = jnp.squeeze(Iagg_devs)
            Yagg_devs = jnp.squeeze(Yagg_devs)

        # Create dictionary with new series
        return {
            "Kagg": Kagg_devs,
            "Cagg_prices": Cagg_devs,
            "Magg": Magg_devs,
            "Iagg": Iagg_devs,
            "Yagg": Yagg_devs,
        }

    def aggregator_tornqvist(
        self,
        simul_obs,
        simul_policy,
        base_period=0,
        use_loop=False,
        add_steady_state=False,
    ):
        """
        This function aggregate K, C, M, I, Y using a Tornqvist index.

        Input:
        simul_obs has dimension (T_simul, 3*n_sectors) or (3*n_sectors,).
        simul_policy has dimension (T_simul, 11*n_sectors+5) or (11*n_sectors+5,).

        Process:
        Extract quantities and prices. Aggregate using Tornqvist formula.

        Output:
        Dictionary with new series for Kagg_tornqvist, Cagg_tornqvist, Magg_tornqvist, Iagg_tornqvist, Yagg_tornqvist.
        """

        # Choose implementation
        if use_loop:
            tornqvist_func = self.tornqvist_index_loop
        else:
            tornqvist_func = self.tornqvist_index_debug

        # Optionally add steady state initial observation
        if add_steady_state:
            print("Adding steady state initial observation")
            ss_obs = jnp.zeros(3 * self.model.n_sectors)
            ss_policy = jnp.ones(11 * self.model.n_sectors + 5)

            # Concatenate steady state with simulation data
            obs_with_ss = jnp.vstack([ss_obs[None, :], simul_obs])
            policy_with_ss = jnp.vstack([ss_policy[None, :], simul_policy])

            # Use the extended data
            obs_to_use = obs_with_ss
            policy_to_use = policy_with_ss
        else:
            obs_to_use = simul_obs
            policy_to_use = simul_policy

        # Process obs and policy - note: assuming time is first dimension
        obs_notnorm = obs_to_use * self.model.obs_sd + self.model.obs_ss  # denormalize
        policy_notnorm = policy_to_use * jnp.exp(self.model.policies_ss)

        # Extract variables (T_simul, n_sectors) or (T_simul+1, n_sectors) if steady state added
        K = jnp.exp(obs_notnorm[:, : self.model.n_sectors])
        C = policy_notnorm[:, : self.model.n_sectors]
        M = policy_notnorm[:, 4 * self.model.n_sectors : 5 * self.model.n_sectors]
        I = policy_notnorm[:, 6 * self.model.n_sectors : 7 * self.model.n_sectors]
        Y = policy_notnorm[:, 10 * self.model.n_sectors : 11 * self.model.n_sectors]

        # Extract prices
        P = policy_notnorm[:, 8 * self.model.n_sectors : 9 * self.model.n_sectors]
        Pk = policy_notnorm[:, 2 * self.model.n_sectors : 3 * self.model.n_sectors]
        Pm = policy_notnorm[:, 3 * self.model.n_sectors : 4 * self.model.n_sectors]

        # Calculate Tornqvist indices for each variable
        Kagg_tornqvist = tornqvist_func(K, Pk, base_period)
        Cagg_tornqvist = tornqvist_func(C, P, base_period)
        Magg_tornqvist = tornqvist_func(M, Pm, base_period)
        Iagg_tornqvist = tornqvist_func(I, Pk, base_period)
        Yagg_tornqvist = tornqvist_func(Y, P, base_period)

        # If steady state was added, remove the first period from results
        if add_steady_state:
            Kagg_tornqvist = Kagg_tornqvist[1:]
            Cagg_tornqvist = Cagg_tornqvist[1:]
            Magg_tornqvist = Magg_tornqvist[1:]
            Iagg_tornqvist = Iagg_tornqvist[1:]
            Yagg_tornqvist = Yagg_tornqvist[1:]

        # Create dictionary with new series
        return {
            "Kagg_tornqvist": Kagg_tornqvist,
            "Cagg_tornqvist": Cagg_tornqvist,
            "Magg_tornqvist": Magg_tornqvist,
            "Iagg_tornqvist": Iagg_tornqvist,
            "Yagg_tornqvist": Yagg_tornqvist,
        }

    def tornqvist_index_debug(self, quantities, prices, base_period=0, verbose=False):
        """
        Debug version of Tornqvist index with detailed logging.
        """
        if verbose:
            print(f"Input shapes - quantities: {quantities.shape}, prices: {prices.shape}")

        # Calculate nominal values
        nominal_values = quantities * prices
        if verbose:
            print(f"Nominal values shape: {nominal_values.shape}")
            print(f"Nominal values sample (t=0): {nominal_values[0, :3]}")

        # Calculate value shares for each period
        total_values = jnp.sum(nominal_values, axis=1, keepdims=True)
        shares = nominal_values / total_values
        if verbose:
            print(f"Total values shape: {total_values.shape}")
            print(f"Shares shape: {shares.shape}")
            print(f"Shares sum (t=0): {jnp.sum(shares[0, :]):.6f}")  # Should be 1.0

        # Calculate log growth rates of quantities
        log_q_growth = jnp.log(quantities[1:, :] / quantities[:-1, :])
        if verbose:
            print(f"Log growth shape: {log_q_growth.shape}")
            print(f"Log growth sample (t=1): {log_q_growth[0, :3]}")

        # Calculate average shares between consecutive periods
        avg_shares = 0.5 * (shares[1:, :] + shares[:-1, :])
        if verbose:
            print(f"Avg shares shape: {avg_shares.shape}")
            print(f"Avg shares sum (t=1): {jnp.sum(avg_shares[0, :]):.6f}")  # Should be ~1.0

        # Calculate weighted log growth
        weighted_log_growth = jnp.sum(avg_shares * log_q_growth, axis=1)
        if verbose:
            print(f"Weighted log growth shape: {weighted_log_growth.shape}")
            print(f"Weighted log growth sample: {weighted_log_growth[:5]}")

        # Calculate cumulative index
        log_index = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(weighted_log_growth)])
        index = jnp.exp(log_index)
        if verbose:
            print(f"Log index shape: {log_index.shape}")
            print(f"Index shape: {index.shape}")
            print(f"Index first 5 values: {index[:5]}")

        # Normalize to base period
        result = index / index[base_period]
        if verbose:
            print(f"Final result shape: {result.shape}")
            print(f"Base period value: {index[base_period]:.6f}")

        return result

    def tornqvist_index_loop(self, quantities, prices, base_period=0):
        """
        Loop-based implementation of Tornqvist index for comparison.
        This follows the traditional step-by-step approach.
        """
        T, N = quantities.shape
        index = jnp.ones(T)  # Initialize index

        for t in range(1, T):
            # Calculate nominal values for periods t-1 and t
            nom_values_prev = quantities[t - 1, :] * prices[t - 1, :]
            nom_values_curr = quantities[t, :] * prices[t, :]

            # Calculate value shares
            total_prev = jnp.sum(nom_values_prev)
            total_curr = jnp.sum(nom_values_curr)
            shares_prev = nom_values_prev / total_prev
            shares_curr = nom_values_curr / total_curr

            # Average shares
            avg_shares = 0.5 * (shares_prev + shares_curr)

            # Quantity relatives
            q_relatives = quantities[t, :] / quantities[t - 1, :]

            # Weighted geometric mean
            log_relatives = jnp.log(q_relatives)
            weighted_log_growth = jnp.sum(avg_shares * log_relatives)

            # Update index
            index = index.at[t].set(index[t - 1] * jnp.exp(weighted_log_growth))

        # Normalize to base period
        return index / index[base_period]
