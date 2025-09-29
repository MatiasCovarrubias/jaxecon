import jax
import jax.numpy as jnp
import jax.random as random


class Model:
    """A JAX implementation of an RBC model."""

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
        self.Cagg_ss = jnp.exp(self.policies_ss[11 * self.n_sectors])
        self.Lagg_ss = jnp.exp(self.policies_ss[11 * self.n_sectors + 1])
        self.utility_ss = (1 / (1 - self.eps_c ** (-1))) * (
            self.Cagg_ss - self.theta * (1 / (1 + self.eps_l ** (-1))) * self.Lagg_ss ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))

    def initial_state(self, rng, range=1):
        """Get initial state by sampling uniformly from a range around the steady state"""

        rng_k, rng_a, rng_c = random.split(rng, 3)
        K_ss = jnp.exp(self.state_ss[: self.n_sectors])  # get K in StSt (in levels)
        A_ss = jnp.exp(self.state_ss[self.n_sectors :])  # get A in StSt (in levels)
        K_init = random.uniform(
            rng_k,
            shape=(self.n_sectors,),
            minval=(1 - range / 100) * K_ss,
            maxval=(1 + range / 300) * K_ss,
        )
        A_init = random.uniform(
            rng_a, shape=(self.n_sectors,), minval=(1 - range / 100) * A_ss, maxval=(1 + range / 100) * A_ss
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
        policy_next_levels = jnp.exp(policy_next_notnorm)
        Pk_next = policy_next_levels[2 * self.n_sectors : 3 * self.n_sectors]
        I_next = policy_next_levels[6 * self.n_sectors : 7 * self.n_sectors]
        P_next = policy_next_levels[8 * self.n_sectors : 9 * self.n_sectors]
        Q_next = policy_next_levels[9 * self.n_sectors : 10 * self.n_sectors]
        Y_next = policy_next_levels[10 * self.n_sectors : 11 * self.n_sectors]

        # Solve for the expectation term in the FOC for Ktplus1
        expect_realization = P_next * A_next ** ((self.sigma_y - 1) / self.sigma_y) * (self.mu * Q_next / Y_next) ** (
            1 / self.sigma_q
        ) * (self.alpha * Y_next / K_next) ** (1 / self.sigma_y) + Pk_next * (
            (1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2)
        )

        return expect_realization

    def loss(self, state, expect, policy):
        """Calculate equilibrium condition losses for given state, policy, and expectations."""

        state_notnorm = state * self.state_sd + self.state_ss  # denormalize
        K = jnp.exp(state_notnorm[: self.n_sectors])  # put in levels
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policy_notnorm = policy * self.policies_sd + self.policies_ss  # denormalize policy
        policy_levels = jnp.exp(policy_notnorm)
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
        Cagg = policy_levels[11 * self.n_sectors]
        Lagg = policy_levels[11 * self.n_sectors + 1]
        Yagg = policy_levels[11 * self.n_sectors + 2]
        Iagg = policy_levels[11 * self.n_sectors + 3]
        Magg = policy_levels[11 * self.n_sectors + 4]

        # get steady state prices to aggregate Y, I and M
        Pss = jnp.exp(self.policies_ss[8 * self.n_sectors : 9 * self.n_sectors])
        Pkss = jnp.exp(self.policies_ss[2 * self.n_sectors : 3 * self.n_sectors])
        Pmss = jnp.exp(self.policies_ss[3 * self.n_sectors : 4 * self.n_sectors])
        capadj_term = 1 - self.phi * (Inv / K - self.delta)

        # auxialiry variables
        MgUtCagg = (Cagg - self.theta * 1 / (1 + self.eps_l ** (-1)) * Lagg ** (1 + self.eps_l ** (-1))) ** (
            -self.eps_c ** (-1)
        )

        # key variables for loss function
        MgUtCmod = MgUtCagg * (Cagg * self.xi / C) ** (1 / self.sigma_c)
        MgUtLmod = MgUtCagg * self.theta * Lagg ** (self.eps_l**-1) * (L / Lagg) ** (1 / self.sigma_l)
        MPLmod = (
            P
            * A ** ((self.sigma_y - 1) / self.sigma_y)
            * (self.mu * Q / Y) ** (1 / self.sigma_q)
            * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
        )
        MPKmod = self.beta * expect
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
        Caggdef = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        Laggdef = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))
        Yaggdef = jnp.sum(Y * Pss)
        Iaggdef = jnp.sum(Inv * Pkss)
        Maggdef = jnp.sum(M * Pmss)

        C_loss = P / MgUtCmod - 1
        L_loss = MgUtLmod / MPLmod - 1
        K_loss = Pk / MPKmod - 1
        Pm_loss = Pm / Pmdef - 1
        M_loss = M / Mmod - 1
        Mout_loss = Mout / Moutmod - 1
        Pk_loss = Pk / Pkdef - 1
        Iout_loss = Iout / Ioutmod - 1
        Qrc_loss = Q / Qrc - 1
        Qdef_loss = Q / Qdef - 1
        Ydef_loss = Y / Ydef - 1
        Caggdef_loss = jnp.array([Cagg / Caggdef - 1])
        Laggdef_loss = jnp.array([Lagg / Laggdef - 1])
        Yaggdef_loss = jnp.array([Yagg / Yaggdef - 1])
        Iaggdef_loss = jnp.array([Iagg / Iaggdef - 1])
        Maggdef_loss = jnp.array([Magg / Maggdef - 1])

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
                Caggdef_loss,
                Laggdef_loss,
                Yaggdef_loss,
                Iaggdef_loss,
                Maggdef_loss,
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
                jnp.mean(1 - jnp.abs(Caggdef_loss)),
                jnp.mean(1 - jnp.abs(Laggdef_loss)),
                jnp.mean(1 - jnp.abs(Yaggdef_loss)),
                jnp.mean(1 - jnp.abs(Iaggdef_loss)),
                jnp.mean(1 - jnp.abs(Maggdef_loss)),
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
                jnp.min(1 - jnp.abs(Caggdef_loss)),
                jnp.min(1 - jnp.abs(Laggdef_loss)),
                jnp.min(1 - jnp.abs(Yaggdef_loss)),
                jnp.min(1 - jnp.abs(Iaggdef_loss)),
                jnp.min(1 - jnp.abs(Maggdef_loss)),
            ]
        )

        return mean_loss, mean_accuracy, min_accuracy, mean_accuracies_focs, min_accuracies_focs

    def get_aggregates(self, state_logdev, policies_logdev, P_weights, Pk_weights, Pm_weights):
        """Calculate log deviations of aggregates from steady state"""
        # Denormalize weights from log deviations to levels
        # Get steady state prices in levels
        P_ss = jnp.exp(self.policies_ss[8 * self.n_sectors : 9 * self.n_sectors])
        Pk_ss = jnp.exp(self.policies_ss[2 * self.n_sectors : 3 * self.n_sectors])
        Pm_ss = jnp.exp(self.policies_ss[3 * self.n_sectors : 4 * self.n_sectors])

        # Convert weights from log deviations to levels
        P_weights_levels = P_ss * jnp.exp(P_weights)
        Pk_weights_levels = Pk_ss * jnp.exp(Pk_weights)
        Pm_weights_levels = Pm_ss * jnp.exp(Pm_weights)

        # Calculate current period aggregates in levels

        Cagg_logdev = policies_logdev[11 * self.n_sectors]
        Lagg_logdev = policies_logdev[11 * self.n_sectors + 1]
        # denormalize policy
        policies_notnorm = policies_logdev + self.policies_ss
        policies_levels = jnp.exp(policies_notnorm)
        Cagg = policies_levels[11 * self.n_sectors]
        Lagg = policies_levels[11 * self.n_sectors + 1]

        # Get Kagg
        state_notnorm = state_logdev + self.state_ss  # denormalize state
        K = jnp.exp(state_notnorm[: self.n_sectors])  # put in levels
        Kagg = K @ Pk_weights_levels

        Y = policies_levels[10 * self.n_sectors : 11 * self.n_sectors]
        Yagg = Y @ P_weights_levels

        M = policies_levels[4 * self.n_sectors : 5 * self.n_sectors]
        Magg = M @ Pm_weights_levels

        Inv = policies_levels[6 * self.n_sectors : 7 * self.n_sectors]
        Iagg = Inv @ Pk_weights_levels

        utility = (1 / (1 - self.eps_c ** (-1))) * (
            Cagg - self.theta * (1 / (1 + self.eps_l ** (-1))) * Lagg ** (1 + self.eps_l ** (-1))
        ) ** (1 - self.eps_c ** (-1))

        # Calculate steady state aggregates in levels using steady state weights (which are just the steady state prices)
        policies_ss_levels = jnp.exp(self.policies_ss)
        Cagg_ss = policies_ss_levels[11 * self.n_sectors]
        Lagg_ss = policies_ss_levels[11 * self.n_sectors + 1]

        # Get steady state Kagg
        K_ss = jnp.exp(self.state_ss[: self.n_sectors])  # put in levels
        Kagg_ss = K_ss @ Pk_weights_levels

        Y_ss = policies_ss_levels[10 * self.n_sectors : 11 * self.n_sectors]
        Yagg_ss = Y_ss @ P_weights_levels

        M_ss = policies_ss_levels[4 * self.n_sectors : 5 * self.n_sectors]
        Magg_ss = M_ss @ Pm_weights_levels

        Inv_ss = policies_ss_levels[6 * self.n_sectors : 7 * self.n_sectors]
        Iagg_ss = Inv_ss @ Pk_weights_levels

        # Calculate log deviations from steady state
        Kagg_logdev = jnp.log(Kagg) - jnp.log(Kagg_ss)
        Yagg_logdev = jnp.log(Yagg) - jnp.log(Yagg_ss)
        Magg_logdev = jnp.log(Magg) - jnp.log(Magg_ss)
        Iagg_logdev = jnp.log(Iagg) - jnp.log(Iagg_ss)
        utility_logdev = 1 - utility / self.utility_ss

        aggregates_array = jnp.array(
            [Cagg_logdev, Lagg_logdev, Kagg_logdev, Yagg_logdev, Magg_logdev, Iagg_logdev, utility_logdev, utility]
        )

        return aggregates_array

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
        # Print row sums
        row_sums = jnp.sum(Delta_M, axis=1)
        print("Row sums of Delta_M:", row_sums)
        # Print column sums
        col_sums = jnp.sum(Delta_M, axis=0)
        print("Column sums of Delta_M:", col_sums)

        # Calculate Delta^I matrix (investment flow upstreamness)
        # Delta^I = Gamma_I * [(P^(-sigma_I)) * (Pk^sigma_I)] * [1_N * (I * Q^(-1))']
        P_term_I = jnp.outer(P ** (-self.sigma_I), Pk**self.sigma_I)
        I_Q_term = jnp.outer(1 / Q, Inv)
        Delta_I = self.Gamma_I * P_term_I * I_Q_term
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
        upstreamness_data = {"sectors": self.labels, "U_M": U_M, "U_I": U_I, "U_simple": U_simple}

        return upstreamness_data
