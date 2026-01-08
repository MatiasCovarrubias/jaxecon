from .rbc_prod_net import RbcProdNet
from .rbc_prod_net_auxiliary import RbcProdNetAuxiliary
from .rbc_prod_net_aggregators import RbcProdNetAggregators
from .rbc_prod_net_tests import RbcProdNetTests


class RbcProdNetComplete:
    """Complete RBC Production Network model combining all components."""

    def __init__(
        self,
        modparams,
        k_ss,
        policies_ss,
        states_sd,
        shocks_sd,
        policies_sd,
        A,
        B,
        C,
        D,
        precision=None,
    ):
        # Initialize base model
        self.model = RbcProdNet(
            modparams,
            k_ss,
            policies_ss,
            states_sd,
            shocks_sd,
            policies_sd,
            A,
            B,
            C,
            D,
            precision,
        )

        # Initialize components
        self.auxiliary = RbcProdNetAuxiliary(self.model)
        self.aggregators = RbcProdNetAggregators(self.model)
        self.tests = RbcProdNetTests(self.model, self.aggregators)

    def __getattr__(self, name):
        """Delegate method calls to appropriate component"""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        elif hasattr(self.auxiliary, name):
            return getattr(self.auxiliary, name)
        elif hasattr(self.aggregators, name):
            return getattr(self.aggregators, name)
        elif hasattr(self.tests, name):
            return getattr(self.tests, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    # Expose key attributes from base model for convenience
    @property
    def alpha(self):
        return self.model.alpha

    @property
    def beta(self):
        return self.model.beta

    @property
    def delta(self):
        return self.model.delta

    @property
    def rho(self):
        return self.model.rho

    @property
    def eps_c(self):
        return self.model.eps_c

    @property
    def eps_l(self):
        return self.model.eps_l

    @property
    def phi(self):
        return self.model.phi

    @property
    def theta(self):
        return self.model.theta

    @property
    def sigma_c(self):
        return self.model.sigma_c

    @property
    def sigma_m(self):
        return self.model.sigma_m

    @property
    def sigma_q(self):
        return self.model.sigma_q

    @property
    def sigma_y(self):
        return self.model.sigma_y

    @property
    def sigma_I(self):
        return self.model.sigma_I

    @property
    def sigma_l(self):
        return self.model.sigma_l

    @property
    def xi(self):
        return self.model.xi

    @property
    def mu(self):
        return self.model.mu

    @property
    def Gamma_M(self):
        return self.model.Gamma_M

    @property
    def Gamma_I(self):
        return self.model.Gamma_I

    @property
    def Sigma_A(self):
        return self.model.Sigma_A

    @property
    def n_sectors(self):
        return self.model.n_sectors

    @property
    def obs_ss(self):
        return self.model.obs_ss

    @property
    def state_ss(self):
        return self.model.state_ss

    @property
    def policies_ss(self):
        return self.model.policies_ss

    @property
    def A(self):
        return self.model.A

    @property
    def B(self):
        return self.model.B

    @property
    def C(self):
        return self.model.C

    @property
    def D(self):
        return self.model.D

    @property
    def obs_sd(self):
        return self.model.obs_sd

    @property
    def shocks_sd(self):
        return self.model.shocks_sd

    @property
    def states_sd(self):
        return self.model.states_sd

    @property
    def dim_policies(self):
        return self.model.dim_policies

    @property
    def n_actions(self):
        return self.model.n_actions

    @property
    def dim_obs(self):
        return self.model.dim_obs

    @property
    def dim_shock(self):
        return self.model.dim_shock

    @property
    def labels(self):
        return self.model.labels
