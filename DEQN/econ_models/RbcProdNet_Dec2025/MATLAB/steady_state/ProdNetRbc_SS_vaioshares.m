function [fx,StStval] = ProdNetRbc_SS_vaioshares(sol,params, print)

n_sectors = params.n_sectors;

%% -------------------- Extra endogenous variables -------------------- %%

mu         = exp(sol(11*n_sectors+3:12*n_sectors+2));
Gamma_M_partial    = exp(sol(12*n_sectors+3:12*n_sectors+2+n_sectors*(n_sectors-1)));
Gamma_M = zeros(n_sectors, n_sectors);
for i = 1:n_sectors
    Gamma_M(1:n_sectors-1, i) = Gamma_M_partial((i-1)*(n_sectors-1)+1 : i*(n_sectors-1));
end
Gamma_M(n_sectors, :) = 1 - sum(Gamma_M(1:n_sectors-1, :), 1);

%% -------------------- Core model equations -------------------- %%

[base_losses, StStval, econ] = ProdNetRbc_SS_core( ...
    sol(1:11*n_sectors+2), params, ...
    params.alpha, mu, params.xi, Gamma_M, params.Gamma_I, print);

%% -------------------- Moment matching -------------------- %%

sigma_q = params.sigma_q;
sigma_m = params.sigma_m;
vashare_data = params.vash_data;
ionet_data = params.ionet_data;

va_share = mu.^(sigma_q^(-1)).*(econ.Ydef./econ.Qdef).^(1-sigma_q^(-1));
Pm_inverted = 1 ./ econ.Pm;
ratio = (econ.P * Pm_inverted.').^(1 - sigma_m);
ionet = Gamma_M .* ratio;

mu_loss = va_share(1:n_sectors)./vashare_data(1:n_sectors) - 1;
Gamma_M_loss = ionet(1:n_sectors-1,:)./ionet_data(1:n_sectors-1,:)-1;
Gamma_M_loss = Gamma_M_loss(:);

%% -------------------- Assemble loss vector -------------------- %%

fx = [base_losses; mu_loss; Gamma_M_loss];

end
