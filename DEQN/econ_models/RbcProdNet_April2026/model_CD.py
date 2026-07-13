import jax.numpy as jnp

try:
    from model import Model as BaseModel
except ImportError:
    from .model import Model as BaseModel


class Model(BaseModel):
    """Exact Cobb-Douglas variant of the April 2026 production-network model."""

    @staticmethod
    def _geomean(weights, values):
        return jnp.exp(weights.T @ jnp.log(values))

    def expect_realization(self, state_next, policy_next):
        """A realization of the expectation terms under exact Cobb-Douglas production."""
        state_next_notnorm = state_next * self.state_sd + self.state_ss
        K_next = jnp.exp(state_next_notnorm[: self.n_sectors])

        policy_next_notnorm = policy_next * self.policies_sd + self.policies_ss
        policy_next_levels = self._policy_levels(policy_next_notnorm)
        Pk_next = policy_next_levels[2 * self.n_sectors : 3 * self.n_sectors]
        I_next = policy_next_levels[6 * self.n_sectors : 7 * self.n_sectors]
        P_next = policy_next_levels[8 * self.n_sectors : 9 * self.n_sectors]
        Q_next = policy_next_levels[9 * self.n_sectors : 10 * self.n_sectors]
        Cagg_next = policy_next_levels[self.c_util_idx]
        Lagg_next = policy_next_levels[self.l_util_idx]
        MU_next = self.marginal_utility(Cagg_next, Lagg_next)

        capital_payoff = P_next * self.mu * self.alpha * Q_next / K_next + Pk_next * (
            (1 - self.delta) + self.phi / 2 * (I_next**2 / K_next**2 - self.delta**2)
        )

        return MU_next * capital_payoff

    def loss(self, state, expect, policy):
        """Calculate equilibrium condition losses using exact Cobb-Douglas limits."""
        state_notnorm = state * self.state_sd + self.state_ss
        K = jnp.exp(state_notnorm[: self.n_sectors])
        a = state_notnorm[self.n_sectors :]
        A = jnp.exp(a)

        policy_notnorm = policy * self.policies_sd + self.policies_ss
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
        utility_intratemp = policy_levels[self.utility_intratemp_idx]

        Pss = jnp.exp(self.policies_ss[8 * self.n_sectors : 9 * self.n_sectors])
        Pkss = jnp.exp(self.policies_ss[2 * self.n_sectors : 3 * self.n_sectors])
        capadj_term = 1 - self.phi * (Inv / K - self.delta)

        MU_t = self.marginal_utility(c_util, l_util)

        Pmod = c_util * self.xi / C
        labor_supply = self.theta * l_util ** (self.eps_l**-1) * (L / l_util) ** (1 / self.sigma_l)
        MPLmod = P * self.mu * (1 - self.alpha) * Q / L
        MPKmod = self.beta * expect / MU_t
        Pmdef = self._geomean(self.Gamma_M, P)
        Mmod = (1 - self.mu) * P * Q / Pm
        Moutmod = P**-1 * jnp.dot(self.Gamma_M, Pm * M)
        Pkdef = self._geomean(self.Gamma_I, P) * capadj_term**-1
        Ioutmod = P**-1 * jnp.dot(self.Gamma_I, Pk * Inv * capadj_term)
        Qrc = C + Mout + Iout
        Qdef = A * jnp.exp(self.mu * jnp.log(Y) + (1 - self.mu) * jnp.log(M))
        Ydef = jnp.exp(self.alpha * jnp.log(K) + (1 - self.alpha) * jnp.log(L))
        c_util_def = jnp.exp(self.xi.T @ jnp.log(C))
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
        utility_intratemp_loss = jnp.array([utility_intratemp / utility_intratemp_def - 1])

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

    def upstreamness(self):
        """Calculate upstreamness under exact Cobb-Douglas network flow shares."""
        policies_ss = jnp.exp(self.policies_ss)
        Pk = policies_ss[2 * self.n_sectors : 3 * self.n_sectors]
        Pm = policies_ss[3 * self.n_sectors : 4 * self.n_sectors]
        M = policies_ss[4 * self.n_sectors : 5 * self.n_sectors]
        Inv = policies_ss[6 * self.n_sectors : 7 * self.n_sectors]
        P = policies_ss[8 * self.n_sectors : 9 * self.n_sectors]
        Q = policies_ss[9 * self.n_sectors : 10 * self.n_sectors]

        identity = jnp.eye(self.n_sectors)
        ones = jnp.ones(self.n_sectors)

        P_term_M = jnp.outer(P**-1, Pm)
        M_Q_term = jnp.outer(1 / Q, M)
        Delta_M = self.Gamma_M * P_term_M * M_Q_term

        P_term_I = jnp.outer(P**-1, Pk)
        I_Q_term = jnp.outer(1 / Q, Inv)
        Delta_I = self.Gamma_I * P_term_I * I_Q_term

        U_M = jnp.linalg.solve(identity - Delta_M, ones)
        U_I = jnp.linalg.solve(identity - Delta_I, ones)
        U_simple = policies_ss[5 * self.n_sectors : 6 * self.n_sectors] / Q

        return {"sectors": self.labels, "U_M": U_M, "U_I": U_I, "U_simple": U_simple}
