import jax
import jax.numpy as jnp
import jax.random as random


class Model:
    """A JAX implementation of an RBC model."""

    def __init__(self, mod_params, k_ss, policies_ss, states_sd, shocks_sd, A, B, C, D, precision):

        # Model parameters
        param_names = [
            "alpha",
            "beta",
            "delta",
            "rho",
            "eps_c",
            "eps_l",
            "phi",
            "theta",
            "sigma_c",
            "sigma_m",
            "sigma_q",
            "sigma_y",
            "sigma_I",
            "sigma_l",
            "xi",
            "mu",
            "Gamma_M",
            "Gamma_I",
            "Sigma_A",
        ]
        for param in param_names:
            setattr(self, param, jnp.array(mod_params[f"par{param}"], dtype=precision))
        self.n_sectors = mod_params["parn_sectors"]
        # State space variables
        self.obs_ss = jnp.concatenate([k_ss, jnp.zeros(shape=(2 * self.n_sectors,), dtype=precision)])
        self.state_ss = jnp.concatenate([k_ss, jnp.zeros(shape=(1 * self.n_sectors,), dtype=precision)])
        self.policies_ss = jnp.array(policies_ss, dtype=precision)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.obs_sd = jnp.concatenate([states_sd, shocks_sd])
        self.shocks_sd = jnp.array(shocks_sd, dtype=precision)
        self.states_sd = jnp.array(states_sd, dtype=precision)
        self.dim_policies = len(policies_ss)
        self.n_actions = self.dim_policies
        self.dim_obs = len(self.obs_ss)
        self.dim_shock = len(shocks_sd)

        # Pre-computed constants to avoid repeating heavy ops at run-time
        self.precision = precision  # dtype used everywhere

        # Exponential of steady-state (log) policies – reused in several methods
        self.policies_ss_level = jnp.exp(self.policies_ss)

        # Steady-state prices (slices of the cached exponential)
        self.Pss = self.policies_ss_level[8 * self.n_sectors : 9 * self.n_sectors]
        self.Pkss = self.policies_ss_level[2 * self.n_sectors : 3 * self.n_sectors]
        self.Pmss = self.policies_ss_level[3 * self.n_sectors : 4 * self.n_sectors]

        # Cached transposed input-output matrices
        self.Gamma_M_T = self.Gamma_M.T
        self.Gamma_I_T = self.Gamma_I.T

        # Scalers needed by the utility function
        self.C_ss = self.policies_ss_level[11 * self.n_sectors]
        self.L_ss = self.policies_ss_level[11 * self.n_sectors + 1]

        # Cholesky factor of the shock covariance for fast shock generation
        self.chol_Sigma_A = jnp.linalg.cholesky(self.Sigma_A)

    def initial_obs(self, rng, range=0):
        """Get initial obs given first shock"""

        rng_k, rng_a, rng_c = random.split(rng, 3)
        e = self.sample_shock(rng)  # sample a realization of the shockt
        K_init = random.uniform(
            rng_k,
            shape=(self.n_sectors,),
            minval=(1 - range / 100) * jnp.exp(self.obs_ss[: self.n_sectors]),
            maxval=(1 + range / 100) * jnp.exp(self.obs_ss[: self.n_sectors]),
        )
        A_init = random.uniform(rng_a, shape=(self.n_sectors,), minval=(1 - range / 100), maxval=(1 + range / 100))
        obs_init_notnorm = jnp.concatenate([jnp.log(K_init), jnp.log(A_init), e])
        obs_init = (obs_init_notnorm - self.obs_ss) / self.obs_sd  # normalize
        return random.choice(rng_c, jnp.array([obs_init, jnp.zeros_like(self.obs_ss)]))

    def step(self, obs, policy, shock):
        """A period step of the model, given current obs, the shock and policy_params"""

        # Get needed variables from obs and policy
        K, a_tmin1, shock_tmin1 = self.get_obs_vars(obs)
        policy_notnorm = policy * self.policies_ss_level
        I = policy_notnorm[6 * self.n_sectors : 7 * self.n_sectors]

        # Update state variables
        a = self.rho * a_tmin1 + shock_tmin1
        K_tplus1 = (1 - self.delta) * K + I - (self.phi / 2) * (I / K - self.delta) ** 2 * K

        # Construct and normalize next observation
        obs_next_notnorm = jnp.concatenate([jnp.log(K_tplus1), a, shock])
        obs_next = (obs_next_notnorm - self.obs_ss) / self.obs_sd

        return obs_next

    def expect_realization(self, obs_next, policy_next):
        """A realization (given a shock) of the expectation terms in system of equation"""

        # Get needed variables from obs and policy
        K_next, a_tmin1, shock_tmin1 = self.get_obs_vars(obs_next)
        A_next = jnp.exp(self.rho * a_tmin1 + shock_tmin1)
        _, _, Pk_next, _, _, _, I_next, _, P_next, Q_next, Y_next, _, _, _, _, _ = self.get_policy_vars(policy_next)

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization = P_next * A_next ** ((self.sigma_y - 1) / self.sigma_y) * (self.mu * Q_next / Y_next) ** (
            1 / self.sigma_q
        ) * (self.alpha * Y_next / K_next) ** (1 / self.sigma_y) + Pk_next * (
            (1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2)
        )

        return expect_realization

    def loss(self, obs, expect, policy):
        """Calculate loss associated with observing obs, having policy_params, and expectation exp"""
        K, a_tmin1, shock_tmin1 = self.get_obs_vars(obs)
        A = jnp.exp(self.rho * a_tmin1 + shock_tmin1)
        (C, L, Pk, Pm, M, Mout, I, Iout, P, Q, Y, Cagg, Lagg, Yagg, Iagg, Magg) = self.get_policy_vars(policy)

        # Auxiliary terms
        MgUtCagg = (Cagg - self.theta * 1 / (1 + self.eps_l ** (-1)) * Lagg ** (1 + self.eps_l ** (-1))) ** (
            -self.eps_c ** (-1)
        )
        capadj_term = 1 - self.phi * (I / K - self.delta)

        # Key equilibrium conditions
        MgUtCmod = MgUtCagg * (Cagg * self.xi / C) ** (1 / self.sigma_c)
        MgUtLmod = MgUtCagg * self.theta * Lagg ** (self.eps_l**-1) * (L / Lagg) ** (1 / self.sigma_l)
        MPLmod = (
            P
            * A ** ((self.sigma_y - 1) / self.sigma_y)
            * (self.mu * Q / Y) ** (1 / self.sigma_q)
            * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
        )
        MPKmod = self.beta * expect
        Pmdef = (self.Gamma_M_T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Mmod = (1 - self.mu) * (Pm / P) ** (-self.sigma_q) * Q
        Moutmod = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        Pkdef = (self.Gamma_I_T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)
        Ioutmod = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * I * capadj_term ** (self.sigma_I))
        Qrc = C + Mout + Iout
        Qdef = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))
        Ydef = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        Caggdef = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Laggdef = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))
        Yaggdef = jnp.sum(Y * self.Pss)
        Iaggdef = jnp.sum(I * self.Pkss)
        Maggdef = jnp.sum(M * self.Pmss)

        # Sectoral equilibrium conditions
        sectoral_losses = (
            P / MgUtCmod - 1,  # 1. C_loss: consumption FOC
            MgUtLmod / MPLmod - 1,  # 2. L_loss: labor FOC
            Pk / MPKmod - 1,  # 3. K_loss: capital FOC
            Pm / Pmdef - 1,  # 4. Pm_loss: materials price definition
            M / Mmod - 1,  # 5. M_loss: materials demand
            Mout / Moutmod - 1,  # 6. Mout_loss: materials market clearing
            Pk / Pkdef - 1,  # 7. Pk_loss: investment price definition
            Iout / Ioutmod - 1,  # 8. Iout_loss: investment market clearing
            Q / Qrc - 1,  # 9. Qrc_loss: resource constraint
            Q / Qdef - 1,  # 10. Qdef_loss: Q definition
            Y / Ydef - 1,  # 11. Ydef_loss: production function
        )

        # Aggregate equilibrium conditions
        aggregate_losses = (
            jnp.array([Cagg / Caggdef - 1]),  # 12. Caggdef_loss: aggregate consumption
            jnp.array([Lagg / Laggdef - 1]),  # 13. Laggdef_loss: aggregate labor
            jnp.array([Yagg / Yaggdef - 1]),  # 14. Yaggdef_loss: aggregate output
            jnp.array([Iagg / Iaggdef - 1]),  # 15. Iaggdef_loss: aggregate investment
            jnp.array([Magg / Maggdef - 1]),  # 16. Maggdef_loss: aggregate materials
        )

        # Combine sectoral and aggregate losses
        losses_tuple = sectoral_losses + aggregate_losses
        accuracies_tuple = jax.tree.map(lambda x: 1 - jnp.abs(x), losses_tuple)
        losses = jnp.concatenate(losses_tuple, axis=0)
        accuracies = jnp.concatenate(accuracies_tuple, axis=0)

        # Aggregate metrics
        mean_loss = jnp.mean(losses**2)
        mean_accuracy = jnp.mean(accuracies)
        min_accuracy = jnp.min(accuracies)
        mean_accuracies_focs = jax.tree.map(jnp.mean, accuracies_tuple)
        min_accuracies_focs = jax.tree.map(jnp.min, accuracies_tuple)

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def sample_shock(self, rng):
        """sample one realization of the shock"""
        z = random.normal(rng, (self.n_sectors,), dtype=self.precision)
        return jnp.dot(self.chol_Sigma_A, z)

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """sample mc_draws realizations of the shock (for monte-carlo)"""
        z = random.normal(rng, (mc_draws, self.n_sectors), dtype=self.precision)
        return jnp.dot(z, self.chol_Sigma_A.T)

    def get_obs_vars(self, obs):
        """Denormalize obs and extract variables. Input: (n,). Output: (K, a, shock)"""
        obs_notnorm = obs * self.obs_sd + self.obs_ss

        K = jnp.exp(obs_notnorm[: self.n_sectors])  # Capital (levels)
        a_tmin1 = obs_notnorm[self.n_sectors : 2 * self.n_sectors]  # Productivity (logs)
        shock_tmin1 = obs_notnorm[2 * self.n_sectors :]  # Shock

        return K, a_tmin1, shock_tmin1

    def get_policy_vars(self, policies):
        """Denormalize policies and extract variables"""
        policy_notnorm = policies * self.policies_ss_level
        n = self.n_sectors

        # Split into sector-specific blocks and aggregate variables
        *blocks, aggs = jnp.split(
            policy_notnorm, [n, 2 * n, 3 * n, 4 * n, 5 * n, 6 * n, 7 * n, 8 * n, 9 * n, 10 * n, 11 * n]
        )
        C, L, Pk, Pm, M, Mout, I, Iout, P, Q, Y = blocks
        Cagg, Lagg, Yagg, Iagg, Magg = aggs[0], aggs[1], aggs[2], aggs[3], aggs[4]

        return (C, L, Pk, Pm, M, Mout, I, Iout, P, Q, Y, Cagg, Lagg, Yagg, Iagg, Magg)

    def step_loglinear(self, obs, shock):
        obs_dev = obs * self.obs_sd
        state_dev = obs_dev[: 2 * self.n_sectors]
        shock_tmin1 = obs_dev[2 * self.n_sectors :]
        new_state_dev = jnp.dot(self.A, state_dev) + jnp.dot(self.B, shock_tmin1)
        new_state = new_state_dev / self.states_sd
        shock_norm = shock / self.shocks_sd
        obs_next = jnp.concatenate([new_state[: 2 * self.n_sectors], shock_norm])
        return obs_next

    def policy_loglinear(self, obs):
        obs_dev = obs * self.obs_sd
        state_dev = obs_dev[: 2 * self.n_sectors]
        shock_tmin1 = obs_dev[2 * self.n_sectors :]
        policy_devs = jnp.dot(self.C, state_dev) + jnp.dot(self.D, shock_tmin1)
        policy_norm = jnp.exp(policy_devs)
        return policy_norm

    def utility(self, C, L):
        C_notnorm = C * self.C_ss
        L_notnorm = L * self.L_ss
        U = (1 / (1 - self.eps_c ** (-1))) * (
            C_notnorm - self.theta * (1 / (1 + self.eps_l ** (-1))) * L_notnorm ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))
        return U

    def ir_shocks(self):
        """Define a set of shocks sequences that are of interest"""
        ir_shock_1 = jnp.zeros(shape=(40, self.n_sectors), dtype=self.precision).at[0, 0].set(-1)
        ir_shock_2 = jnp.zeros(shape=(40, self.n_sectors), dtype=self.precision).at[0, 0].set(1)

        return jnp.array([ir_shock_1, ir_shock_2])

    def get_vars_with_aux(self, obs, policies):
        """Get auxiliary variables that depend on basic variables."""
        # Get basic variables
        K, A = self.get_obs_vars(obs)
        (C, L, Pk, Pm, M, Mout, I, Iout, P, Q, Y, Cagg, Lagg, Yagg, Iagg, Magg) = self.get_policy_vars(policies)

        # Calculate auxiliary variables
        capadj_term = 1 - self.phi * (I / K - self.delta)
        Pagg = (self.xi.T @ P ** (1 - self.sigma_c)) ** (1 / (1 - self.sigma_c))
        utility = self.utility(Cagg, Lagg)

        vars_tuple = (K, A, C, L, Pk, Pm, M, Mout, I, Iout, P, Q, Y, Cagg, Lagg, Yagg, Iagg, Magg)
        aux_tuple = (capadj_term, Pagg, utility)

        return vars_tuple, aux_tuple
