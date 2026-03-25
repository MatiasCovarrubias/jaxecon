import jax.numpy as jnp

try:
    from model_debug import ModelDebug as VAModelDebug, check_nan, check_positive
except ImportError:
    from .model_debug import ModelDebug as VAModelDebug, check_nan, check_positive


class ModelDebug(VAModelDebug):
    """Debug GO-shock variant of the RBC model with production networks."""

    def expect_realization(self, state_next, policy_next):
        state_next_notnorm = state_next * self.state_sd + self.state_ss
        K_next = jnp.exp(state_next_notnorm[: self.n_sectors])
        a_next = state_next_notnorm[self.n_sectors : 2 * self.n_sectors]
        A_next = jnp.exp(a_next)

        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss
        policy_next_levels = jnp.exp(policy_next_notnorm)

        check_nan(policy_next_notnorm, "policy_next_notnorm", "expect_realization")
        check_nan(policy_next_levels, "policy_next_levels", "expect_realization")

        Pk_next = policy_next_levels[2 * self.n_sectors : 3 * self.n_sectors]
        I_next = policy_next_levels[6 * self.n_sectors : 7 * self.n_sectors]
        P_next = policy_next_levels[8 * self.n_sectors : 9 * self.n_sectors]
        Q_next = policy_next_levels[9 * self.n_sectors : 10 * self.n_sectors]
        Y_next = policy_next_levels[10 * self.n_sectors : 11 * self.n_sectors]
        Cagg_next = policy_next_levels[11 * self.n_sectors]
        Lagg_next = policy_next_levels[11 * self.n_sectors + 1]
        MU_next = self.marginal_utility(Cagg_next, Lagg_next)

        check_positive(Pk_next, "Pk_next", "expect_realization")
        check_positive(I_next, "I_next", "expect_realization")
        check_positive(P_next, "P_next", "expect_realization")
        check_positive(Q_next, "Q_next", "expect_realization")
        check_positive(Y_next, "Y_next", "expect_realization")
        check_positive(K_next, "K_next", "expect_realization")
        check_positive(A_next, "A_next", "expect_realization")

        term1_base = self.mu * Q_next / Y_next
        check_positive(term1_base, "mu*Q/Y", "expect_realization term1")

        term2_base = self.alpha * Y_next / K_next
        check_positive(term2_base, "alpha*Y/K", "expect_realization term2")

        A_power = A_next ** ((self.sigma_q - 1) / self.sigma_q)
        check_nan(A_power, "A^((sigma_q-1)/sigma_q)", "expect_realization")

        Q_Y_power = term1_base ** (1 / self.sigma_q)
        check_nan(Q_Y_power, "(mu*Q/Y)^(1/sigma_q)", "expect_realization")

        Y_K_power = term2_base ** (1 / self.sigma_y)
        check_nan(Y_K_power, "(alpha*Y/K)^(1/sigma_y)", "expect_realization")

        expect_term1 = P_next * A_power * Q_Y_power * Y_K_power
        check_nan(expect_term1, "expect_term1 (MPK-related)", "expect_realization")

        I_K_term = I_next**2 / K_next**2 - self.delta**2
        expect_term2 = Pk_next * ((1 - self.delta) + self.phi / 2 * I_K_term)
        check_nan(expect_term2, "expect_term2 (Pk adjustment)", "expect_realization")

        capital_payoff = expect_term1 + expect_term2
        expect_realization = MU_next * capital_payoff
        check_nan(expect_realization, "expect_realization (final)", "expect_realization")

        return expect_realization

    def loss(self, state, expect, policy):
        check_nan(expect, "expect (input)", "loss function input")

        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policy_notnorm = policy * self.policies_sd + self.policies_ss
        policy_levels = jnp.exp(policy_notnorm)

        check_nan(policy_levels, "policy_levels", "loss")

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

        check_positive(C, "C", "loss - policy extraction")
        check_positive(L, "L", "loss - policy extraction")
        check_positive(Cagg, "Cagg", "loss - policy extraction")
        check_positive(Lagg, "Lagg", "loss - policy extraction")

        Pss = jnp.exp(self.policies_ss[8 * self.n_sectors : 9 * self.n_sectors])
        Pkss = jnp.exp(self.policies_ss[2 * self.n_sectors : 3 * self.n_sectors])
        Pmss = jnp.exp(self.policies_ss[3 * self.n_sectors : 4 * self.n_sectors])
        capadj_term = 1 - self.phi * (Inv / K - self.delta)

        MgUtCagg_inner = Cagg - self.theta * 1 / (1 + self.eps_l ** (-1)) * Lagg ** (1 + self.eps_l ** (-1))
        check_positive(MgUtCagg_inner, "MgUtCagg_inner (Cagg - theta*...)", "loss")

        MU_t = MgUtCagg_inner ** (-self.eps_c ** (-1))
        check_nan(MU_t, "MU_t", "loss")

        Pmod = (Cagg * self.xi / C) ** (1 / self.sigma_c)
        check_nan(Pmod, "Pmod", "loss")

        labor_supply = self.theta * Lagg ** (self.eps_l**-1) * (L / Lagg) ** (1 / self.sigma_l)
        check_nan(labor_supply, "labor_supply", "loss")

        MPLmod = (
            P
            * A ** ((self.sigma_q - 1) / self.sigma_q)
            * (self.mu * Q / Y) ** (1 / self.sigma_q)
            * ((1 - self.alpha) * Y / L) ** (1 / self.sigma_y)
        )
        check_nan(MPLmod, "MPLmod", "loss")

        MPKmod = self.beta * expect / MU_t
        check_nan(MPKmod, "MPKmod", "loss")

        Pmdef = (self.Gamma_M.T @ P ** (1 - self.sigma_m)) ** (1 / (1 - self.sigma_m))
        check_nan(Pmdef, "Pmdef", "loss")

        Mmod = (1 - self.mu) * (Pm / (A ** ((self.sigma_q - 1) / self.sigma_q) * P)) ** (-self.sigma_q) * Q
        check_nan(Mmod, "Mmod", "loss")

        Moutmod = P ** (-self.sigma_m) * jnp.dot(self.Gamma_M, Pm**self.sigma_m * M)
        check_nan(Moutmod, "Moutmod", "loss")

        Pkdef = (self.Gamma_I.T @ P ** (1 - self.sigma_I)) ** (1 / (1 - self.sigma_I)) * capadj_term ** (-1)
        check_nan(Pkdef, "Pkdef", "loss")

        Ioutmod = P ** (-self.sigma_I) * jnp.dot(self.Gamma_I, Pk**self.sigma_I * Inv * capadj_term ** (self.sigma_I))
        check_nan(Ioutmod, "Ioutmod", "loss")

        Qrc = C + Mout + Iout
        Qdef = A * (
            self.mu ** (1 / self.sigma_q) * Y ** ((self.sigma_q - 1) / self.sigma_q)
            + (1 - self.mu) ** (1 / self.sigma_q) * M ** ((self.sigma_q - 1) / self.sigma_q)
        ) ** (self.sigma_q / (self.sigma_q - 1))
        check_nan(Qdef, "Qdef", "loss")

        Ydef = (
            self.alpha ** (1 / self.sigma_y) * K ** ((self.sigma_y - 1) / self.sigma_y)
            + (1 - self.alpha) ** (1 / self.sigma_y) * L ** ((self.sigma_y - 1) / self.sigma_y)
        ) ** (self.sigma_y / (self.sigma_y - 1))
        check_nan(Ydef, "Ydef", "loss")

        Caggdef = ((self.xi ** (1 / self.sigma_c)).T @ C ** ((self.sigma_c - 1) / self.sigma_c)) ** (
            self.sigma_c / (self.sigma_c - 1)
        )
        check_nan(Caggdef, "Caggdef", "loss")

        Laggdef = jnp.sum(L ** ((self.sigma_l + 1) / self.sigma_l)) ** (self.sigma_l / (self.sigma_l + 1))
        check_nan(Laggdef, "Laggdef", "loss")

        Yaggdef = jnp.sum(Y * Pss)
        Iaggdef = jnp.sum(Inv * Pkss)
        Maggdef = jnp.sum(M * Pmss)

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

        check_nan(losses_array, "losses_array", "loss - final")

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
