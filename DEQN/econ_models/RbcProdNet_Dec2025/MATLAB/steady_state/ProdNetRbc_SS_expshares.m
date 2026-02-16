function [fx,ModData] = ProdNetRbc_SS_expshares(sol,params, print)

n_sectors = params.n_sectors;

%% -------------------- Extra endogenous variables -------------------- %%

xi         = exp(sol(11*n_sectors+3:12*n_sectors+2));
mu         = exp(sol(12*n_sectors+3:13*n_sectors+2));
alpha         = exp(sol(13*n_sectors+3:14*n_sectors+2));

Gamma_M_partial    = exp(sol(14*n_sectors+3:14*n_sectors+2+n_sectors*(n_sectors-1)));
Gamma_M = zeros(n_sectors, n_sectors);
for i = 1:n_sectors
    Gamma_M(1:n_sectors-1, i) = Gamma_M_partial((i-1)*(n_sectors-1)+1 : i*(n_sectors-1));
end
Gamma_M(n_sectors, :) = 1 - sum(Gamma_M(1:n_sectors-1, :), 1);

Gamma_I_partial    = exp(sol(14*n_sectors+2+n_sectors*(n_sectors-1)+1:14*n_sectors+2+n_sectors*(n_sectors-1)+n_sectors*(n_sectors-1)));
Gamma_I = zeros(n_sectors, n_sectors);
for i = 1:n_sectors
    Gamma_I(1:n_sectors-1, i) = Gamma_I_partial((i-1)*(n_sectors-1)+1 : i*(n_sectors-1));
end
Gamma_I(n_sectors, :) = 1 - sum(Gamma_I(1:n_sectors-1, :), 1);

%% -------------------- Core model equations -------------------- %%

[base_losses, ModData, econ] = ProdNetRbc_SS_core( ...
    sol(1:11*n_sectors+2), params, ...
    alpha, mu, xi, Gamma_M, Gamma_I, print);

%% -------------------- Moment matching -------------------- %%

sigma_c = params.sigma_c;
sigma_q = params.sigma_q;
sigma_y = params.sigma_y;
sigma_m = params.sigma_m;
sigma_I = params.sigma_I;
consshare_data = params.conssh_data;
vashare_data = params.vash_data;
capshare_data = params.capsh_data;
ionet_data = params.ionet_data;
invnet_data = params.invnet_data;

cons_share = xi.^(sigma_c^(-1)).*(econ.C/econ.Cagg).^(1-sigma_c^(-1));
va_share = mu.^(sigma_q^(-1)).*(econ.Ydef./econ.Qdef).^(1-sigma_q^(-1));
cap_share = alpha.^(sigma_y^(-1)).*(econ.K./econ.Ydef).^(1-sigma_y^(-1));
Pm_inverted = 1 ./ econ.Pm;
ratio = (econ.P * Pm_inverted.').^(1 - sigma_m);
ionet = Gamma_M .* ratio;
Pk_inverted = 1 ./ econ.Pk;
ratio = (econ.P * Pk_inverted.').^(1 - sigma_I);
invnet = Gamma_I .* ratio;

xi_loss = cons_share./consshare_data-1;
mu_loss = va_share./vashare_data - 1;
alpha_loss = cap_share./capshare_data - 1;
Gamma_M_loss = ionet(1:n_sectors-1,:)./ionet_data(1:n_sectors-1,:)-1;
Gamma_M_loss = Gamma_M_loss(:);
Gamma_I_loss = invnet(1:n_sectors-1,:)./invnet_data(1:n_sectors-1,:)-1;
Gamma_I_loss = Gamma_I_loss(:);

%% -------------------- Assemble loss vector -------------------- %%

fx = [base_losses; xi_loss; mu_loss; alpha_loss; Gamma_M_loss; Gamma_I_loss];
ModData.loss = fx;

end
