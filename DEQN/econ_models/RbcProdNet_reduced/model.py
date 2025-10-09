import jax
import jax.numpy as jnp
import jax.random as random


class Model:
    """Reduced RBC model with production networks using 4N policy layout [L, M, Inv, P]."""

    def __init__(
        self,
        parameters,
        state_ss,
        policies_ss,
        state_sd,
        policies_sd,
        double_precision=True,
        volatility_scale=1,
    ):
        # Precision
        precision = jnp.float64 if double_precision else jnp.float32

        self.volatility_scale = jnp.array(volatility_scale, dtype=precision)

        # Parameters
        self.alpha = jnp.array(parameters["paralpha"], dtype=precision)
        self.beta = jnp.array(parameters["parbeta"], dtype=precision)
        self.delta = jnp.array(parameters["pardelta"], dtype=precision)
        self.rho = jnp.array(parameters["parrho"], dtype=precision)
        self.eps_c = jnp.array(parameters["pareps_c"], dtype=precision)
        self.eps_l = jnp.array(parameters["pareps_l"], dtype=precision)
        self.phi = jnp.array(parameters["parphi"], dtype=precision)
        self.theta = jnp.array(parameters["partheta"], dtype=precision)
        self.sigma_c = jnp.array(parameters["parsigma_c"], dtype=precision)
        self.sigma_m = jnp.array(parameters["parsigma_m"], dtype=precision)
        self.sigma_q = jnp.array(parameters["parsigma_q"], dtype=precision)
        self.sigma_y = jnp.array(parameters["parsigma_y"], dtype=precision)
        self.sigma_I = jnp.array(parameters["parsigma_I"], dtype=precision)
        self.sigma_l = jnp.array(parameters["parsigma_l"], dtype=precision)
        self.xi = jnp.array(parameters["parxi"], dtype=precision)
        self.mu = jnp.array(parameters["parmu"], dtype=precision)
        self.Gamma_M = jnp.array(parameters["parGamma_M"], dtype=precision)
        self.Gamma_I = jnp.array(parameters["parGamma_I"], dtype=precision)
        self.Sigma_A = jnp.array(parameters["parSigma_A"], dtype=precision)
        self.n_sectors = parameters["parn_sectors"]

        # Steady states and scales (log space for policies and states)
        self.state_ss = jnp.array(state_ss, dtype=precision)
        self.policies_ss = jnp.array(policies_ss, dtype=precision)
        self.state_sd = jnp.array(state_sd, dtype=precision)
        self.policies_sd = jnp.array(policies_sd, dtype=precision)
        self.dim_policies = len(self.policies_ss)
        self.dim_states = len(self.state_ss)
        self.L_cholesky = jnp.linalg.cholesky(self.Sigma_A)

        # Labels (copied for analysis convenience)
        self.labels = [
            "Mining, Oil and Gas",
            "Utilities",
            "Construction",
            "Wood",
            "Minerals",
            "Primary Metals",
            "Fabricated Metals",
            "Machinery",
            "Computers",
            "Electrical",
            "Vehicles",
            "Transport",
            "Furniture",
            "Misc Mfg",
            "Food Mfg",
            "Textile",
            "Apparel",
            "Paper",
            "Printing",
            "Petroleum",
            "Chemical",
            "Plastics",
            "Wholesale Trade",
            "Retail",
            "Transp. and Wareh.",
            "Info",
            "Finance",
            "Real estate",
            "Prof/Tech",
            "Mgmt",
            "Admin",
            "Educ",
            "Health",
            "Arts",
            "Accom",
            "Food Services",
            "Other",
        ]

    def initial_state(self, rng, range=1):
        rng_k, rng_a, rng_c = random.split(rng, 3)
        K_ss = jnp.exp(self.state_ss[: self.n_sectors])
        A_ss = jnp.exp(self.state_ss[self.n_sectors :])
        K_init = random.uniform(
            rng_k, shape=(self.n_sectors,), minval=(1 - range / 100) * K_ss, maxval=(1 + range / 300) * K_ss
        )
        A_init = random.uniform(
            rng_a, shape=(self.n_sectors,), minval=(1 - range / 100) * A_ss, maxval=(1 + range / 100) * A_ss
        )
        state_init_notnorm = jnp.concatenate([jnp.log(K_init), jnp.log(A_init)])
        state_init = (state_init_notnorm - self.state_ss) / self.state_sd
        return random.choice(rng_c, jnp.array([state_init]))

    def step(self, state, policy, shock):
        """State transition using reduced policy vector [L, M, Inv, P]."""
        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        a_next = self.rho * a + shock

        policy_notnorm = policy * self.policies_sd + self.policies_ss
        Inv = jnp.exp(policy_notnorm[2 * self.n_sectors : 3 * self.n_sectors])

        K_tplus1 = (1 - self.delta) * K + Inv - (self.phi / 2) * (Inv / K - self.delta) ** 2 * K
        state_next_notnorm = jnp.concatenate([jnp.log(K_tplus1), a_next])
        state_next = (state_next_notnorm - self.state_ss) / self.state_sd
        return state_next

    def expect_realization(self, state_next, policy_next):
        """Expectation realization using reduced policy vector [L, M, Inv, P]."""
        state_next_notnorm = state_next * self.state_sd + self.state_ss
        K_next = jnp.exp(state_next_notnorm[: self.n_sectors])
        a_next = state_next_notnorm[self.n_sectors : 2 * self.n_sectors]
        A_next = jnp.exp(a_next)

        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss
        policy_next_levels = jnp.exp(policy_next_notnorm)

        L_next = policy_next_levels[0 : self.n_sectors]
        M_next = policy_next_levels[self.n_sectors : 2 * self.n_sectors]
        I_next = policy_next_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P_next = policy_next_levels[3 * self.n_sectors : 4 * self.n_sectors]

        capadj_term_next = 1 - self.phi * (I_next / K_next - self.delta)
        Pk_next = (self.Gamma_I.T @ P_next ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term_next ** (-1)

        Y_next = A_next * (
            self.alpha ** (1 / self.sigma_y) * K_next ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L_next ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))

        Q_next = (
            self.mu ** (1 / self.sigma_q) * Y_next ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M_next ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))

        expect_realization = P_next * A_next ** ((self.sigma_y - 1) / self.sigma_y) * (self.mu * Q_next / Y_next) ** (
            1 / self.sigma_q
        ) * (self.alpha * Y_next / K_next) ** (1 / self.sigma_y) + Pk_next * (
            (1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2)
        )

        return expect_realization

    def loss(self, state, expect, policy):
        """Reduced system loss using L, M, Inv, P as endogenous variables."""
        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policy_notnorm = policy * self.policies_sd + self.policies_ss
        policy_levels = jnp.exp(policy_notnorm)

        L = policy_levels[0 : self.n_sectors]
        M = policy_levels[self.n_sectors : 2 * self.n_sectors]
        Inv = policy_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P = policy_levels[3 * self.n_sectors : 4 * self.n_sectors]

        capadj_term = 1 - self.phi * (Inv / K - self.delta)
        Pm = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Pk = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)

        Mout = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        Iout = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * Inv * capadj_term ** (self.sigma_I))

        Y = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))

        Q = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))

        C = Q - Mout - Iout

        Cagg = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Lagg = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))

        MgUtCagg = (Cagg - self.theta * 1 / (1 + self.eps_l ** (-1)) * Lagg ** (1 + self.eps_l ** (-1))) ** (
            -self.eps_c ** (-1)
        )
        MgUtCmod_temp = MgUtCagg * (Cagg * self.xi / C) ** (1 / self.sigma_c)
        normC = (self.xi.T @ MgUtCmod_temp ** (1 - self.sigma_c)) ** (1 / (1 - self.sigma_c))
        MgUtCmod = MgUtCmod_temp / normC

        MgUtLmod = MgUtCagg * self.theta * Lagg ** (self.eps_l**-1) * (L / Lagg) ** (1 / self.sigma_l) / normC

        MPLmod = (
            P
            * A ** ((self.sigma_y - 1) / self.sigma_y)
            * (self.mu * Q / Y) ** (1 / self.sigma_q)
            * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
        )

        MPKmod = self.beta * expect
        Pm_P_ratio = Pm / P
        Mmod = (1 - self.mu) * Pm_P_ratio ** (-self.sigma_q) * Q

        C_loss = P / MgUtCmod - 1
        L_loss = MgUtLmod / MPLmod - 1
        K_loss = Pk / MPKmod - 1
        M_loss = M / Mmod - 1

        losses_array = jnp.concatenate([C_loss, L_loss, K_loss, M_loss], axis=0)
        mean_loss = jnp.mean(losses_array**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(losses_array))
        min_accuracy = jnp.min(1 - jnp.abs(losses_array))
        mean_accuracies_focs = jnp.array(
            [
                jnp.mean(1 - jnp.abs(C_loss)),
                jnp.mean(1 - jnp.abs(L_loss)),
                jnp.mean(1 - jnp.abs(K_loss)),
                jnp.mean(1 - jnp.abs(M_loss)),
            ]
        )
        min_accuracies_focs = jnp.array(
            [
                jnp.min(1 - jnp.abs(C_loss)),
                jnp.min(1 - jnp.abs(L_loss)),
                jnp.min(1 - jnp.abs(K_loss)),
                jnp.min(1 - jnp.abs(M_loss)),
            ]
        )

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def utility_from_policies(self, policies_logdev, state_logdev=None):
        """Utility using reduced policy vector [L, M, Inv, P]. Inputs are log deviations.

        Note: For the reduced model, we need state_logdev to compute intermediate variables.
        If state_logdev is None, we assume policies are at steady state (used for compatibility).
        """
        if state_logdev is None:
            # Assume at steady state - use zero log-deviations
            state_logdev = jnp.zeros_like(self.state_ss)

        state_notnorm = state_logdev + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policies_notnorm = policies_logdev + self.policies_ss
        policies_levels = jnp.exp(policies_notnorm)
        L = policies_levels[0 : self.n_sectors]
        M = policies_levels[self.n_sectors : 2 * self.n_sectors]
        Inv = policies_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P = policies_levels[3 * self.n_sectors : 4 * self.n_sectors]

        capadj_term = 1 - self.phi * (Inv / K - self.delta)
        Pm = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Pk = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)
        Mout = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        Iout = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * Inv * capadj_term ** (self.sigma_I))

        Y = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        Q = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))
        C = Q - Mout - Iout

        Cagg = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Lagg = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))

        utility = (1 / (1 - self.eps_c ** (-1))) * (
            Cagg - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))
        return utility

    def get_analysis_variables(self, state_logdev, policies_logdev, P_weights, Pk_weights, Pm_weights):
        """Analysis variables for reduced 4N policy layout [L, M, Inv, P]."""
        state_notnorm = state_logdev + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policies_notnorm = policies_logdev + self.policies_ss
        policies_levels = jnp.exp(policies_notnorm)

        L = policies_levels[0 : self.n_sectors]
        M = policies_levels[self.n_sectors : 2 * self.n_sectors]
        Inv = policies_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P = policies_levels[3 * self.n_sectors : 4 * self.n_sectors]

        # Construct steady state prices from policies_ss (which contains [L, M, Inv, P])
        policies_ss_levels = jnp.exp(self.policies_ss)
        L_ss = policies_ss_levels[0 : self.n_sectors]
        M_ss = policies_ss_levels[self.n_sectors : 2 * self.n_sectors]
        Inv_ss = policies_ss_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P_ss = policies_ss_levels[3 * self.n_sectors : 4 * self.n_sectors]

        # Compute steady state price aggregates
        K_ss_temp = jnp.exp(self.state_ss[: self.n_sectors])
        capadj_term_ss_temp = 1 - self.phi * (Inv_ss / K_ss_temp - self.delta)
        Pm_ss_temp = (self.Gamma_M.T @ P_ss ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Pk_ss_temp = (self.Gamma_I.T @ P_ss ** (1 - self.sigma_I)) ** (
            1 / (1 - self.sigma_I)
        ) * capadj_term_ss_temp ** (-1)

        P_weights_levels = P_ss * jnp.exp(P_weights)
        Pk_weights_levels = Pk_ss_temp * jnp.exp(Pk_weights)
        Pm_weights_levels = Pm_ss_temp * jnp.exp(Pm_weights)

        capadj_term = 1 - self.phi * (Inv / K - self.delta)
        Pm = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Pk = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)

        Mout = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        Iout = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * Inv * capadj_term ** (self.sigma_I))

        Y = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        Q = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))

        C = Q - Mout - Iout
        Cagg = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Lagg = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))

        # Compute steady state aggregates from policies_ss (reuse previously computed Pm_ss_temp, Pk_ss_temp)
        Mout_ss = P_ss ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm_ss_temp**self.sigma_m * M_ss)
        Iout_ss = P_ss ** (-self.sigma_I) * jnp.dot(
            self.Gamma_I, Pk_ss_temp**self.sigma_I * Inv_ss * capadj_term_ss_temp ** (self.sigma_I)
        )

        A_ss = jnp.exp(self.state_ss[self.n_sectors :])
        K_ss = K_ss_temp
        Y_ss = A_ss * (
            self.alpha ** (1 / self.sigma_y) * K_ss ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L_ss ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        Q_ss = (
            self.mu ** (1 / self.sigma_q) * Y_ss ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M_ss ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))

        C_ss = Q_ss - Mout_ss - Iout_ss
        Cagg_ss = ((self.xi ** (1 / self.sigma_c)).T @ C_ss ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Lagg_ss = jnp.sum(L_ss ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))

        Kagg = K @ Pk_weights_levels
        Yagg = Y @ P_weights_levels
        Magg = M @ Pm_weights_levels
        Iagg = Inv @ Pk_weights_levels

        Kagg_ss = K_ss @ Pk_weights_levels
        Yagg_ss = Y_ss @ P_weights_levels
        Magg_ss = M_ss @ Pm_weights_levels
        Iagg_ss = Inv_ss @ Pk_weights_levels

        Cagg_logdev = jnp.log(Cagg) - jnp.log(Cagg_ss)
        Lagg_logdev = jnp.log(Lagg) - jnp.log(Lagg_ss)
        Kagg_logdev = jnp.log(Kagg) - jnp.log(Kagg_ss)
        Yagg_logdev = jnp.log(Yagg) - jnp.log(Yagg_ss)
        Magg_logdev = jnp.log(Magg) - jnp.log(Magg_ss)
        Iagg_logdev = jnp.log(Iagg) - jnp.log(Iagg_ss)

        utility = (1 / (1 - self.eps_c ** (-1))) * (
            Cagg - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))
        utility_ss = (1 / (1 - self.eps_c ** (-1))) * (
            Cagg_ss - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg_ss ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))
        utility_dev = (utility - utility_ss) / jnp.abs(utility_ss)

        return {
            "Agg. Consumption": Cagg_logdev,
            "Agg. Labor": Lagg_logdev,
            "Agg. Capital": Kagg_logdev,
            "Agg. Output": Yagg_logdev,
            "Agg. Intermediates": Magg_logdev,
            "Agg. Investment": Iagg_logdev,
            "Utility": utility_dev,
        }

    def sample_shock(self, rng):
        shock = jax.random.multivariate_normal(rng, jnp.zeros((self.n_sectors,)), self.Sigma_A)
        return self.volatility_scale * shock

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        def latin_hypercube_sample(key, n_samples, n_dims):
            keys = random.split(key, n_dims)
            perms = jnp.stack([random.permutation(keys[i], n_samples) for i in range(n_dims)], axis=1)
            key_uniform = random.fold_in(key, 1)
            uniform_noise = random.uniform(key_uniform, shape=(n_samples, n_dims))
            lhs_samples = (perms + uniform_noise) / n_samples
            return lhs_samples

        use_antithetic = mc_draws % 2 == 0
        if use_antithetic:
            n_base = mc_draws // 2
            key1, key2 = random.split(rng)
            u_lhs = latin_hypercube_sample(key1, n_base, self.n_sectors)
            u_lhs = jnp.clip(u_lhs, 1e-6, 1 - 1e-6)
            z_base = jax.scipy.stats.norm.ppf(u_lhs)
            z = jnp.vstack([z_base, -z_base])
        else:
            u_lhs = latin_hypercube_sample(rng, mc_draws, self.n_sectors)
            u_lhs = jnp.clip(u_lhs, 1e-6, 1 - 1e-6)
            z = jax.scipy.stats.norm.ppf(u_lhs)

        if hasattr(self, "L_cholesky"):
            mc_shocks = jax.vmap(lambda zi: self.L_cholesky @ zi)(z)
        else:
            L_cholesky = jnp.linalg.cholesky(self.Sigma_A)
            mc_shocks = jax.vmap(lambda zi: L_cholesky @ zi)(z)

        return self.volatility_scale * mc_shocks

    def upstreamness(self, state_logdev, policies_logdev):
        """Upstreamness using reduced 4N policy layout [L, M, Inv, P]."""
        state_notnorm = state_logdev + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policies_notnorm = policies_logdev + self.policies_ss
        policies_levels = jnp.exp(policies_notnorm)
        L = policies_levels[0 : self.n_sectors]
        M = policies_levels[self.n_sectors : 2 * self.n_sectors]
        Inv = policies_levels[2 * self.n_sectors : 3 * self.n_sectors]
        P = policies_levels[3 * self.n_sectors : 4 * self.n_sectors]

        capadj_term = 1 - self.phi * (Inv / K - self.delta)
        Pm = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Pk = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)

        Y = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        Q = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))

        identity = jnp.eye(self.n_sectors)
        ones = jnp.ones(self.n_sectors)
        P_term_M = jnp.outer(P ** (-self.sigma_m), Pm**self.sigma_m)
        M_Q_term = jnp.outer(1 / Q, M)
        Delta_M = self.Gamma_M * P_term_M * M_Q_term
        P_term_I = jnp.outer(P ** (-self.sigma_I), Pk**self.sigma_I)
        I_Q_term = jnp.outer(1 / Q, Inv)
        Delta_I = self.Gamma_I * P_term_I * I_Q_term
        U_M = jnp.linalg.solve(identity - Delta_M, ones)
        U_I = jnp.linalg.solve(identity - Delta_I, ones)
        U_simple = (P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)) / Q
        return {"sectors": self.labels, "U_M": U_M, "U_I": U_I, "U_simple": U_simple}
