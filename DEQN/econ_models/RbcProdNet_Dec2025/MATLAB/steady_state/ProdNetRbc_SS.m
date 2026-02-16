function [fx,StStval] = ProdNetRbc_SS(sol,params, print)

[fx, StStval] = ProdNetRbc_SS_core(sol, params, ...
    params.alpha, params.mu, params.xi, params.Gamma_M, params.Gamma_I, print);

end
