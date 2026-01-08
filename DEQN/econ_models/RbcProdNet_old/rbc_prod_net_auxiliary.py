from jax import numpy as jnp
from jax import random
import jax


class RbcProdNetAuxiliary:
    """Auxiliary methods for RBC Production Network model analysis."""

    def __init__(self, model):
        """Initialize with reference to base model."""
        self.model = model

    def utility(self, C, L):
        """Calculate utility function."""
        C_notnorm = C * jnp.exp(self.model.policies_ss[11 * self.model.n_sectors])
        L_notnorm = L * jnp.exp(self.model.policies_ss[11 * self.model.n_sectors + 1])
        U = (1 / (1 - self.model.eps_c ** (-1))) * (
            C_notnorm
            - self.model.theta * (1 / (1 + self.model.eps_l ** (-1))) * L_notnorm ** (1 + self.model.eps_l ** (-1))
        ) ** (1 - self.model.eps_c ** (-1))
        return U

    def ir_shocks(self, precision=jnp.float64):
        """(Optional) Define a set of shocks sequences that are of interest"""
        ir_shock_1 = jnp.zeros(shape=(40, self.model.n_sectors), dtype=precision).at[0, 0].set(-1)
        ir_shock_2 = jnp.zeros(shape=(40, self.model.n_sectors), dtype=precision).at[0, 0].set(1)

        return jnp.array([ir_shock_1, ir_shock_2])

    def upstreamness(self):
        """Calculate the upstreamness of each sector based on intermediate inputs and investment flows"""
        # Process policy
        policies_ss = jnp.exp(self.model.policies_ss)
        Pk = policies_ss[2 * self.model.n_sectors : 3 * self.model.n_sectors]
        Pm = policies_ss[3 * self.model.n_sectors : 4 * self.model.n_sectors]
        M = policies_ss[4 * self.model.n_sectors : 5 * self.model.n_sectors]
        Mout = policies_ss[5 * self.model.n_sectors : 6 * self.model.n_sectors]
        I = policies_ss[6 * self.model.n_sectors : 7 * self.model.n_sectors]
        P = policies_ss[8 * self.model.n_sectors : 9 * self.model.n_sectors]
        Q = policies_ss[9 * self.model.n_sectors : 10 * self.model.n_sectors]

        # Create identity matrix
        identity = jnp.eye(self.model.n_sectors)
        ones = jnp.ones(self.model.n_sectors)

        # Calculate Delta^M matrix (intermediate input upstreamness)
        P_term_M = jnp.outer(P ** (-self.model.sigma_m), Pm**self.model.sigma_m)
        M_Q_term = jnp.outer(1 / Q, M)
        Delta_M = self.model.Gamma_M * P_term_M * M_Q_term
        # Print row sums
        row_sums = jnp.sum(Delta_M, axis=1)
        print("Row sums of Delta_M:", row_sums)
        # Print column sums
        col_sums = jnp.sum(Delta_M, axis=0)
        print("Column sums of Delta_M:", col_sums)

        # Calculate Delta^I matrix (investment flow upstreamness)
        # Delta^I = Gamma_I * [(P^(-sigma_I)) * (Pk^sigma_I)] * [1_N * (I * Q^(-1))']
        P_term_I = jnp.outer(P ** (-self.model.sigma_I), Pk**self.model.sigma_I)
        I_Q_term = jnp.outer(1 / Q, I)
        Delta_I = self.model.Gamma_I * P_term_I * I_Q_term
        # Print row sums
        row_sums = jnp.sum(Delta_I, axis=1)
        print("Row sums of Delta_I:", row_sums)
        # Print column sums
        col_sums = jnp.sum(Delta_I, axis=0)
        print("Column sums of Delta_I:", col_sums)

        # Calculate upstreamness measures
        # U^M = [I - Delta^M]^(-1) * 1
        # U^I = [I - Delta^I]^(-1) * 1
        U_M = jnp.linalg.solve(identity - Delta_M, ones)
        U_I = jnp.linalg.solve(identity - Delta_I, ones)

        # Calculate alternative upstreamness measure: Mout/Q
        U_simple = Mout / Q

        # Create a dictionary with sector labels and upstreamness measures
        upstreamness_data = {
            "sectors": self.model.labels,
            "U_M": U_M,
            "U_I": U_I,
            "U_simple": U_simple,
        }

        return upstreamness_data

    def get_variables(self, obs, policies):
        """
        Extract all economic variables from observations and policies.

        Input:
        obs has dimension (T_simul, 3*n_sectors) or (3*n_sectors,).
        policies has dimension (T_simul, 11*n_sectors+5) or (11*n_sectors+5,).

        Process:
        Denormalize obs and policies, then extract all variables.

        Output:
        Dictionary with all economic variables.
        """

        # Handle vector inputs by adding time dimension
        is_vector_input = False
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)  # (3*n_sectors,) -> (1, 3*n_sectors)
            is_vector_input = True
        if policies.ndim == 1:
            policies = policies.reshape(1, -1)  # (11*n_sectors+5,) -> (1, 11*n_sectors+5)

        # Denormalize obs and policies
        obs_notnorm = obs * self.model.obs_sd + self.model.obs_ss
        policy_notnorm = policies * jnp.exp(self.model.policies_ss)

        # Extract variables from obs
        K = jnp.exp(obs_notnorm[:, : self.model.n_sectors])  # Capital (levels)
        a = obs_notnorm[:, self.model.n_sectors : 2 * self.model.n_sectors]  # Productivity (logs)
        shock = obs_notnorm[:, 2 * self.model.n_sectors : 3 * self.model.n_sectors]  # Shocks
        A = jnp.exp(a)  # Productivity (levels)

        # Extract variables from policies
        C = policy_notnorm[:, : self.model.n_sectors]
        L = policy_notnorm[:, self.model.n_sectors : 2 * self.model.n_sectors]
        Pk = policy_notnorm[:, 2 * self.model.n_sectors : 3 * self.model.n_sectors]
        Pm = policy_notnorm[:, 3 * self.model.n_sectors : 4 * self.model.n_sectors]
        M = policy_notnorm[:, 4 * self.model.n_sectors : 5 * self.model.n_sectors]
        Mout = policy_notnorm[:, 5 * self.model.n_sectors : 6 * self.model.n_sectors]
        I = policy_notnorm[:, 6 * self.model.n_sectors : 7 * self.model.n_sectors]
        Iout = policy_notnorm[:, 7 * self.model.n_sectors : 8 * self.model.n_sectors]
        P = policy_notnorm[:, 8 * self.model.n_sectors : 9 * self.model.n_sectors]
        Q = policy_notnorm[:, 9 * self.model.n_sectors : 10 * self.model.n_sectors]
        Y = policy_notnorm[:, 10 * self.model.n_sectors : 11 * self.model.n_sectors]
        Cagg = policy_notnorm[:, 11 * self.model.n_sectors]
        Lagg = policy_notnorm[:, 11 * self.model.n_sectors + 1]
        Yagg = policy_notnorm[:, 11 * self.model.n_sectors + 2]
        Iagg = policy_notnorm[:, 11 * self.model.n_sectors + 3]
        Magg = policy_notnorm[:, 11 * self.model.n_sectors + 4]

        # Create variables dictionary
        variables = {
            # State variables
            "K": K,
            "a": a,
            "A": A,
            "shock": shock,
            # Policy variables
            "C": C,
            "L": L,
            "Pk": Pk,
            "Pm": Pm,
            "M": M,
            "Mout": Mout,
            "I": I,
            "Iout": Iout,
            "P": P,
            "Q": Q,
            "Y": Y,
            "Cagg": Cagg,
            "Lagg": Lagg,
            "Yagg": Yagg,
            "Iagg": Iagg,
            "Magg": Magg,
        }

        # If input was vector, squeeze output to remove time dimension
        if is_vector_input:
            variables = {key: jnp.squeeze(value) for key, value in variables.items()}

        return variables
