import jax
import jax.numpy as jnp
import jax.random as random


class Model:
    """A JAX implementation of an RBC model with production networks."""

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
        # Set precision based on boolean parameter
        precision = jnp.float64 if double_precision else jnp.float32

        self.volatility_scale = jnp.array(volatility_scale, dtype=precision)

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
        self.state_ss = jnp.array(state_ss, dtype=precision)
        self.policies_ss = jnp.array(policies_ss, dtype=precision)

        self.state_sd = jnp.array(state_sd, dtype=precision)
        self.policies_sd = jnp.array(policies_sd, dtype=precision)
        self.dim_policies = len(self.policies_ss)
        self.dim_states = len(self.state_ss)
        self.L_cholesky = jnp.linalg.cholesky(self.Sigma_A)
        n = self.n_sectors
        self.c_util_idx = 11 * n
        self.l_util_idx = 11 * n + 1
        self.c_agg_idx = 11 * n + 2
        self.l_agg_idx = 11 * n + 3
        self.gdp_agg_idx = 11 * n + 4
        self.i_agg_idx = 11 * n + 5
        self.k_agg_idx = 11 * n + 6
        self.utility_intratemp_idx = 11 * n + 7
        self.log_policy_count = self.utility_intratemp_idx
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
        self.c_util_ss = jnp.exp(self.policies_ss[self.c_util_idx])
        self.l_util_ss = jnp.exp(self.policies_ss[self.l_util_idx])
        self.Cagg_ss = self.c_util_ss
        self.Lagg_ss = self.l_util_ss
        self.c_agg_ss = jnp.exp(self.policies_ss[self.c_agg_idx])
        self.l_agg_ss = jnp.exp(self.policies_ss[self.l_agg_idx])
        self.gdp_agg_ss = jnp.exp(self.policies_ss[self.gdp_agg_idx])
        self.i_agg_ss = jnp.exp(self.policies_ss[self.i_agg_idx])
        self.k_agg_ss = jnp.exp(self.policies_ss[self.k_agg_idx])
        self.utility_intratemp_ss = self.policies_ss[self.utility_intratemp_idx]
        self.utility_ss = (1 / (1 - self.eps_c ** (-1))) * (
            self.c_util_ss
            - self.theta * (1 / (1 + self.eps_l ** (-1))) * self.l_util_ss ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))

    def marginal_utility(self, Cagg, Lagg):
        return (Cagg - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg ** (1 + self.eps_l ** (-1))) ** (
            -self.eps_c ** (-1)
        )

    def _policy_levels(self, policy_notnorm):
        return jnp.exp(policy_notnorm[: self.log_policy_count])

    def initial_state(self, rng, init_range=1):
        """Get initial state by sampling uniformly from a range around the steady state.

        Args:
            rng: JAX random key
            init_range: Either a scalar (used for both endo and exo states) or a dict with keys
                       'endostate' and 'exostate' for separate control over endogenous (K)
                       and exogenous (A) state initialization ranges.
                       The range value represents percentage deviation from steady state.
        """
        if isinstance(init_range, dict):
            range_endo = init_range.get("endostate", 1)
            range_exo = init_range.get("exostate", 1)
        else:
            range_endo = init_range
            range_exo = init_range

        rng_k, rng_a, rng_c = random.split(rng, 3)
        K_ss = jnp.exp(self.state_ss[: self.n_sectors])  # get K in StSt (in levels)
        A_ss = jnp.exp(self.state_ss[self.n_sectors :])  # get A in StSt (in levels)
        K_init = random.uniform(
            rng_k,
            shape=(self.n_sectors,),
            minval=(1 - range_endo / 100) * K_ss,
            maxval=(1 + range_endo / 300) * K_ss,
        )
        A_init = random.uniform(
            rng_a, shape=(self.n_sectors,), minval=(1 - range_exo / 100) * A_ss, maxval=(1 + range_exo / 100) * A_ss
        )
        state_init_notnorm = jnp.concatenate([jnp.log(K_init), jnp.log(A_init)])
        state_init = (state_init_notnorm - self.state_ss) / self.state_sd  # normalize
        return random.choice(rng_c, jnp.array([state_init]))

    def step(self, state, policy, shock):
        """A period step of the model, given current state, the shock and policy"""

        state_notnorm = state * self.state_sd + self.state_ss  # denormalize state
        K = jnp.exp(state_notnorm[: self.n_sectors])  # extract k and put in levels
        a = state_notnorm[self.n_sectors :]
        a_next = self.rho * a + shock

        policy_notnorm = policy * self.policies_sd + self.policies_ss  # denormalize policy
        Inv = jnp.exp(policy_notnorm[6 * self.n_sectors : 7 * self.n_sectors])

        K_tplus1 = (1 - self.delta) * K + Inv - (self.phi / 2) * (Inv / K - self.delta) ** 2 * K  # update K
        state_next_notnorm = jnp.concatenate([jnp.log(K_tplus1), a_next])  # calculate next state not notrm
        state_next = (state_next_notnorm - self.state_ss) / self.state_sd  # normalize

        return state_next

    def expect_realization(self, state_next, policy_next):
        """A realization (given a shock) of the expectation terms in system of equation"""

        state_next_notnorm = state_next * self.state_sd + self.state_ss  # denormalize
        K_next = jnp.exp(state_next_notnorm[: self.n_sectors])  # put in levels
        a_next = state_next_notnorm[self.n_sectors : 2 * self.n_sectors]
        A_next = jnp.exp(a_next)

        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss  # denormalize policy
        policy_next_levels = self._policy_levels(policy_next_notnorm)
        Pk_next = policy_next_levels[2 * self.n_sectors : 3 * self.n_sectors]
        I_next = policy_next_levels[6 * self.n_sectors : 7 * self.n_sectors]
        P_next = policy_next_levels[8 * self.n_sectors : 9 * self.n_sectors]
        Q_next = policy_next_levels[9 * self.n_sectors : 10 * self.n_sectors]
        Y_next = policy_next_levels[10 * self.n_sectors : 11 * self.n_sectors]
        Cagg_next = policy_next_levels[self.c_util_idx]
        Lagg_next = policy_next_levels[self.l_util_idx]
        MU_next = self.marginal_utility(Cagg_next, Lagg_next)

        capital_payoff = P_next * A_next ** ((self.sigma_y - 1) / self.sigma_y) * (self.mu * Q_next / Y_next) ** (
            1 / self.sigma_q
        ) * (self.alpha * Y_next / K_next) ** (1 / self.sigma_y) + Pk_next * (
            (1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2)
        )
        expect_realization = MU_next * capital_payoff

        return expect_realization

    def loss(self, state, expect, policy):
        """Calculate equilibrium condition losses for given state, policy, and expectations."""

        state_notnorm = state * self.state_sd + self.state_ss  # denormalize
        K = jnp.exp(state_notnorm[: self.n_sectors])  # put in levels
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policy_notnorm = policy * self.policies_sd + self.policies_ss  # denormalize policy
        policy_levels = self._policy_levels(policy_notnorm)
        C = policy_levels[: self.n_sectors]
        L = policy_levels[self.n_sectors : 2 * self.n_sectors]
        Pk = policy_levels[2 * self.n_sectors : 3 * self.n_sectors]
        Pm = policy_levels[3 * self.n_sectors : 4 * self.n_sectors]
        M = policy_levels[4 * self.n_sectors : 5 * self.n_sectors]
        Mout = policy_levels[5 * self.n_sectors : 6 * self.n_sectors]
        Inv = policy_levels[6 * self.n_sectors : 7 * self.n_sectors]
        Iout = policy_levels[7 * self.n_sectors : 8 * self.n_sectors]
        P = policy_levels[8 * self.n_sectors : 9 * self.n_sectors]
        Q = policy_levels[9 * self.n_sectors : 10 * self.n_sectors]
        Y = policy_levels[10 * self.n_sectors : 11 * self.n_sectors]
        c_util = policy_levels[self.c_util_idx]
        l_util = policy_levels[self.l_util_idx]
        c_agg = policy_levels[self.c_agg_idx]
        l_agg = policy_levels[self.l_agg_idx]
        gdp_agg = policy_levels[self.gdp_agg_idx]
        i_agg = policy_levels[self.i_agg_idx]
        k_agg = policy_levels[self.k_agg_idx]
        utility_intratemp = policy_notnorm[self.utility_intratemp_idx]

        # get steady state prices to aggregate Y, I and M
        Pss = jnp.exp(self.policies_ss[8 * self.n_sectors : 9 * self.n_sectors])
        Pkss = jnp.exp(self.policies_ss[2 * self.n_sectors : 3 * self.n_sectors])
        capadj_term = 1 - self.phi * (Inv / K - self.delta)

        # auxialiry variables
        MU_t = self.marginal_utility(c_util, l_util)

        # key variables for loss function
        Pmod = (c_util * self.xi / C) ** (1 / self.sigma_c)
        labor_supply = self.theta * l_util ** (self.eps_l**-1) * (L / l_util) ** (1 / self.sigma_l)
        MPLmod = (
            P
            * A ** ((self.sigma_y - 1) / self.sigma_y)
            * (self.mu * Q / Y) ** (1 / self.sigma_q)
            * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
        )
        MPKmod = self.beta * expect / MU_t
        Pmdef = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        Mmod = (1 - self.mu) * (Pm / P) ** (-self.sigma_q) * Q
        Moutmod = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        Pkdef = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)
        Ioutmod = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * Inv * capadj_term ** (self.sigma_I))
        Qrc = C + Mout + Iout
        Qdef = (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))
        Ydef = A * (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        c_util_def = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        l_util_def = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))
        c_agg_def = jnp.sum(Pss * C)
        l_agg_def = jnp.sum(L)
        gdp_agg_def = jnp.sum(Pss * (Q - Mout))
        i_agg_def = jnp.sum(Pss * Iout)
        k_agg_def = jnp.sum(Pkss * K)
        utility_intratemp_def = c_util - self.theta * (1 / (1 + self.eps_l ** (-1))) * l_util ** (
            1 + self.eps_l ** (-1)
        )

        C_loss = P / Pmod - 1
        L_loss = labor_supply / MPLmod - 1
        K_loss = Pk / MPKmod - 1
        Pm_loss = Pm / Pmdef - 1
        M_loss = M / Mmod - 1
        Mout_loss = Mout / Moutmod - 1
        Pk_loss = Pk / Pkdef - 1
        Iout_loss = Iout / Ioutmod - 1
        Qrc_loss = Q / Qrc - 1
        Qdef_loss = Q / Qdef - 1
        Ydef_loss = Y / Ydef - 1
        c_util_loss = jnp.array([c_util / c_util_def - 1])
        l_util_loss = jnp.array([l_util / l_util_def - 1])
        c_agg_loss = jnp.array([c_agg / c_agg_def - 1])
        l_agg_loss = jnp.array([l_agg / l_agg_def - 1])
        gdp_agg_loss = jnp.array([gdp_agg / gdp_agg_def - 1])
        i_agg_loss = jnp.array([i_agg / i_agg_def - 1])
        k_agg_loss = jnp.array([k_agg / k_agg_def - 1])
        utility_intratemp_loss = jnp.array([utility_intratemp - utility_intratemp_def])

        losses_array = jnp.concatenate(
            [
                C_loss,
                L_loss,
                K_loss,
                Pm_loss,
                M_loss,
                Mout_loss,
                Pk_loss,
                Iout_loss,
                Qrc_loss,
                Qdef_loss,
                Ydef_loss,
                c_util_loss,
                l_util_loss,
                c_agg_loss,
                l_agg_loss,
                gdp_agg_loss,
                i_agg_loss,
                k_agg_loss,
                utility_intratemp_loss,
            ],
            axis=0,
        )

        # Calculate aggregate losses and metrics
        mean_loss = jnp.mean(losses_array**2)
        mean_accuracy = jnp.mean(1 - jnp.abs(losses_array))
        min_accuracy = jnp.min(1 - jnp.abs(losses_array))
        mean_accuracies_focs = jnp.array(
            [
                jnp.mean(1 - jnp.abs(C_loss)),
                jnp.mean(1 - jnp.abs(L_loss)),
                jnp.mean(1 - jnp.abs(K_loss)),
                jnp.mean(1 - jnp.abs(Pm_loss)),
                jnp.mean(1 - jnp.abs(M_loss)),
                jnp.mean(1 - jnp.abs(Mout_loss)),
                jnp.mean(1 - jnp.abs(Pk_loss)),
                jnp.mean(1 - jnp.abs(Iout_loss)),
                jnp.mean(1 - jnp.abs(Qrc_loss)),
                jnp.mean(1 - jnp.abs(Qdef_loss)),
                jnp.mean(1 - jnp.abs(Ydef_loss)),
                jnp.mean(1 - jnp.abs(c_util_loss)),
                jnp.mean(1 - jnp.abs(l_util_loss)),
                jnp.mean(1 - jnp.abs(c_agg_loss)),
                jnp.mean(1 - jnp.abs(l_agg_loss)),
                jnp.mean(1 - jnp.abs(gdp_agg_loss)),
                jnp.mean(1 - jnp.abs(i_agg_loss)),
                jnp.mean(1 - jnp.abs(k_agg_loss)),
                jnp.mean(1 - jnp.abs(utility_intratemp_loss)),
            ]
        )

        min_accuracies_focs = jnp.array(
            [
                jnp.min(1 - jnp.abs(C_loss)),
                jnp.min(1 - jnp.abs(L_loss)),
                jnp.min(1 - jnp.abs(K_loss)),
                jnp.min(1 - jnp.abs(Pm_loss)),
                jnp.min(1 - jnp.abs(M_loss)),
                jnp.min(1 - jnp.abs(Mout_loss)),
                jnp.min(1 - jnp.abs(Pk_loss)),
                jnp.min(1 - jnp.abs(Iout_loss)),
                jnp.min(1 - jnp.abs(Qrc_loss)),
                jnp.min(1 - jnp.abs(Qdef_loss)),
                jnp.min(1 - jnp.abs(Ydef_loss)),
                jnp.min(1 - jnp.abs(c_util_loss)),
                jnp.min(1 - jnp.abs(l_util_loss)),
                jnp.min(1 - jnp.abs(c_agg_loss)),
                jnp.min(1 - jnp.abs(l_agg_loss)),
                jnp.min(1 - jnp.abs(gdp_agg_loss)),
                jnp.min(1 - jnp.abs(i_agg_loss)),
                jnp.min(1 - jnp.abs(k_agg_loss)),
                jnp.min(1 - jnp.abs(utility_intratemp_loss)),
            ]
        )

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def utility_from_policies(self, policies_logdev):
        """
        Calculate utility from policies.

        Args:
            policies_logdev: Policy variables in log deviation form

        Returns:
            utility: Utility in levels
        """
        # Denormalize policies
        policies_notnorm = policies_logdev + self.policies_ss
        policies_levels = self._policy_levels(policies_notnorm)

        # Extract aggregate consumption and labor
        Cagg = policies_levels[self.c_util_idx]
        Lagg = policies_levels[self.l_util_idx]

        # Calculate utility
        utility = (1 / (1 - self.eps_c ** (-1))) * (
            Cagg - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))

        return utility

    def get_aggregates(self, policies_logdev):
        """Return model-implied aggregate policy variables in log-deviation form."""
        return {
            "Agg. Consumption": policies_logdev[self.c_agg_idx],
            "Agg. Labor": policies_logdev[self.l_agg_idx],
            "Agg. Capital": policies_logdev[self.k_agg_idx],
            "Agg. Output": policies_logdev[self.gdp_agg_idx],
            "Agg. GDP": policies_logdev[self.gdp_agg_idx],
            "Agg. Investment": policies_logdev[self.i_agg_idx],
            "Intratemporal Utility": policies_logdev[self.utility_intratemp_idx],
        }

    def sample_shock(self, rng):
        """sample one realization of the shock"""
        shock = jax.random.multivariate_normal(rng, jnp.zeros((self.n_sectors,)), self.Sigma_A)
        return self.volatility_scale * shock

    def mc_shocks(self, rng=random.PRNGKey(0), mc_draws=8):
        """
        Optimized for highly nonlinear functions using:
        - Antithetic variates (always helps with symmetric distributions)
        - Latin Hypercube Sampling (better than simple stratification for nonlinear cases)
        - Optional importance sampling direction (if you know where nonlinearity is strongest)
        """

        # Latin Hypercube Sampling for better space-filling with nonlinear functions
        def latin_hypercube_sample(key, n_samples, n_dims):
            """Generate Latin Hypercube samples"""
            keys = random.split(key, n_dims)

            # Create permutations for each dimension
            perms = jnp.stack([random.permutation(keys[i], n_samples) for i in range(n_dims)], axis=1)

            # Add uniform noise within each cell
            key_uniform = random.fold_in(key, 1)
            uniform_noise = random.uniform(key_uniform, shape=(n_samples, n_dims))

            # Create LHS samples in [0,1]^d
            lhs_samples = (perms + uniform_noise) / n_samples
            return lhs_samples

        # Decide on sampling strategy
        use_antithetic = mc_draws % 2 == 0

        if use_antithetic:
            # Generate half samples with LHS, create antithetic pairs
            n_base = mc_draws // 2
            key1, key2 = random.split(rng)

            # Latin hypercube sampling for base samples
            u_lhs = latin_hypercube_sample(key1, n_base, self.n_sectors)

            # Transform to standard normal
            u_lhs = jnp.clip(u_lhs, 1e-6, 1 - 1e-6)
            z_base = jax.scipy.stats.norm.ppf(u_lhs)

            # Create antithetic pairs (works well even for nonlinear functions)
            z = jnp.vstack([z_base, -z_base])

        else:
            # Full Latin Hypercube Sampling
            u_lhs = latin_hypercube_sample(rng, mc_draws, self.n_sectors)
            u_lhs = jnp.clip(u_lhs, 1e-6, 1 - 1e-6)
            z = jax.scipy.stats.norm.ppf(u_lhs)

        # Optional: Add controlled noise for highly discontinuous functions
        # This can help explore around discontinuities
        # key_noise = random.fold_in(rng, 2)
        # noise = 0.1 * random.normal(key_noise, shape=z.shape)
        # z = z + noise

        # Transform to target distribution
        if hasattr(self, "L_cholesky"):
            mc_shocks = jax.vmap(lambda zi: self.L_cholesky @ zi)(z)
        else:
            L_cholesky = jnp.linalg.cholesky(self.Sigma_A)
            mc_shocks = jax.vmap(lambda zi: L_cholesky @ zi)(z)

        return self.volatility_scale * mc_shocks

    def utility(self, C, L):
        C_notnorm = C * jnp.exp(self.policies_ss[11 * self.n_sectors])
        L_notnorm = L * jnp.exp(self.policies_ss[11 * self.n_sectors + 1])
        U = (1 / (1 - self.eps_c ** (-1))) * (
            C_notnorm - self.theta * (1 / (1 + self.eps_l ** (-1))) * L_notnorm ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))
        return U

    def consumption_equivalent(self, welfare):
        """
        Calculate consumption-equivalent welfare measure (Vc).

        This inverts the welfare to find what steady-state consumption level would
        deliver the same welfare, expressed as a deviation from actual steady state.

        Args:
            welfare: Discounted welfare V = E[sum_{t=0}^inf beta^t U_t]

        Returns:
            Vc: Consumption equivalent measure.
                Vc = 0 at steady state.
                Vc < 0 means welfare loss (would need to increase C_ss to compensate).
                Vc > 0 means welfare gain.
        """
        labor_disutility_ss = self.theta * (1 / (1 + self.eps_l ** (-1))) * self.Lagg_ss ** (1 + self.eps_l ** (-1))

        sigma = self.eps_c ** (-1)
        exponent = 1 / (1 - sigma)
        X_from_welfare = (welfare * (1 - self.beta) * (1 - sigma)) ** exponent

        Vc = (1 / self.Cagg_ss) * (X_from_welfare + labor_disutility_ss) - 1

        return Vc

    def upstreamness(self):
        """Calculate the upstreamness of each sector based on intermediate inputs and investment flows"""
        # Process policy
        policies_ss = jnp.exp(self.policies_ss)
        Pk = policies_ss[2 * self.n_sectors : 3 * self.n_sectors]
        Pm = policies_ss[3 * self.n_sectors : 4 * self.n_sectors]
        M = policies_ss[4 * self.n_sectors : 5 * self.n_sectors]
        Mout = policies_ss[5 * self.n_sectors : 6 * self.n_sectors]
        Inv = policies_ss[6 * self.n_sectors : 7 * self.n_sectors]
        P = policies_ss[8 * self.n_sectors : 9 * self.n_sectors]
        Q = policies_ss[9 * self.n_sectors : 10 * self.n_sectors]

        # Create identity matrix
        identity = jnp.eye(self.n_sectors)
        ones = jnp.ones(self.n_sectors)

        # Calculate Delta^M matrix (intermediate input upstreamness)
        P_term_M = jnp.outer(P ** (-self.sigma_m), Pm**self.sigma_m)
        M_Q_term = jnp.outer(1 / Q, M)
        Delta_M = self.Gamma_M * P_term_M * M_Q_term

        # Calculate Delta^I matrix (investment flow upstreamness)
        P_term_I = jnp.outer(P ** (-self.sigma_I), Pk**self.sigma_I)
        I_Q_term = jnp.outer(1 / Q, Inv)
        Delta_I = self.Gamma_I * P_term_I * I_Q_term

        # Calculate upstreamness measures
        # U^M = [I - Delta^M]^(-1) * 1
        # U^I = [I - Delta^I]^(-1) * 1
        U_M = jnp.linalg.solve(identity - Delta_M, ones)
        U_I = jnp.linalg.solve(identity - Delta_I, ones)

        # Calculate alternative upstreamness measure: Mout/Q
        U_simple = Mout / Q

        # Create a dictionary with sector labels and upstreamness measures
        upstreamness_data = {"sectors": self.labels, "U_M": U_M, "U_I": U_I, "U_simple": U_simple}

        return upstreamness_data
