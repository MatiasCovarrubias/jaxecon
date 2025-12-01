% Steady State
function [fx,StStval] = ProdNetRbc_SS_vaioshares(sol,params, print)

% This functions computes the steady state of the model

%% -------------------- Parameters -------------------- %%

alpha = params.alpha;
beta = params.beta;
delta = params.delta;
rho = params.rho;
eps_l = params.eps_l;
eps_c = params.eps_c; 
phi = params.phi;
sigma_c = params.sigma_c;
sigma_m = params.sigma_m;
sigma_q = params.sigma_q;
sigma_y = params.sigma_y;
sigma_I = params.sigma_I;
sigma_l = params.sigma_l;
xi = params.xi;
Gamma_I = params.Gamma_I;
n_sectors = params.n_sectors;
Sigma_A = params.Sigma_A;
vashare_data = params.vash_data;
ionet_data = params.ionet_data;


%% -------------------- Endogenous variables -------------------- %%
C          = exp(sol(1:n_sectors));
L          = exp(sol(n_sectors+1:2*n_sectors));
Pk          = exp(sol(2*n_sectors+1:3*n_sectors));
Pm          = exp(sol(3*n_sectors+1:4*n_sectors));
M          = exp(sol(4*n_sectors+1:5*n_sectors));
Mout       = exp(sol(5*n_sectors+1:6*n_sectors));
I          = exp(sol(6*n_sectors+1:7*n_sectors));
Iout       = exp(sol(7*n_sectors+1:8*n_sectors));
P          = exp(sol(8*n_sectors+1:9*n_sectors));
Q          = exp(sol(9*n_sectors+1:10*n_sectors));
Y          = exp(sol(10*n_sectors+1:11*n_sectors));
Cagg       = exp(sol(11*n_sectors+1));
Lagg       = exp(sol(11*n_sectors+2));
theta      = params.theta;

mu         = exp(sol(11*n_sectors+3:12*n_sectors+2));
Gamma_M_partial    = exp(sol(12*n_sectors+3:12*n_sectors+2+n_sectors*(n_sectors-1)));
Gamma_M = zeros(n_sectors, n_sectors);
for i = 1:n_sectors
    Gamma_M(1:n_sectors-1, i) = Gamma_M_partial((i-1)*(n_sectors-1)+1 : i*(n_sectors-1));
end
Gamma_M(n_sectors, :) = 1 - sum(Gamma_M(1:n_sectors-1, :), 1);


%% -------------------- Model equations -------------------- %%

% Economic Variables

K = I./delta;
Pagg = (sum(xi'*P.^(1-sigma_c)))^(1/(1-sigma_c));
MgUtCagg = (Cagg-theta*(1/(1+eps_l^(-1)))*Lagg^(1+eps_l^(-1)))^(-eps_c^(-1));
MgUtCmod_temp = MgUtCagg * (Cagg * xi./C).^(1/sigma_c);
normC = (xi' * MgUtCmod_temp.^(1 - sigma_c))^(1/(1 - sigma_c)); 
MgUtCmod = MgUtCmod_temp/normC;
MgUtLmod = MgUtCagg * theta*Lagg^(eps_l^(-1)) * (L/Lagg).^(1/sigma_l)/normC;
MPLmod = P .* ((mu).*Q./Y).^(1/sigma_q) .* ((1-alpha).*Y./L).^(1/sigma_y);
MPKmod = (beta./(1-beta*(1-delta))).*(P.*((mu).*Q./Y).^(1/sigma_q) .* (alpha.*Y./K).^(1/sigma_y));
Pmdef = (Gamma_M.'* (P.^(1-sigma_m))).^(1/(1-sigma_m)) ;
Mmod = (1-mu) .* (Pm./P).^(-sigma_q) .* Q  ;
Moutmod = P.^(-sigma_m) .* (Gamma_M*((Pm.^(sigma_m)).*M)); 
Pkdef =(Gamma_I.'* (P.^(1-sigma_I))).^(1/(1-sigma_I));
Ioutmod = P.^(-sigma_I) .* (Gamma_I* ((Pk.^(sigma_I)).*I)) ;
Qrc = C+Mout+Iout;
Qdef = (((mu).^(1/sigma_q).*Y.^((sigma_q-1)/sigma_q) + (1-mu).^(1/sigma_q).*M.^((sigma_q-1)/sigma_q) ).^(sigma_q/(sigma_q-1)));
Ydef = ((alpha.^(1/sigma_y).*K.^((sigma_y-1)/sigma_y) + (1-alpha).^(1/sigma_y).*L.^((sigma_y-1)/sigma_y) ).^(sigma_y/(sigma_y-1)));
Caggdef = ( (xi.^(1/sigma_c))' * (C.^((sigma_c-1)/sigma_c)) )^(sigma_c/(sigma_c-1));
Laggdef = (sum(L.^((sigma_l+1)/sigma_l)))^(sigma_l/(sigma_l+1));
V = (1/(1-eps_c^(-1)))*(Cagg-theta*(1/(1+eps_l^(-1)))*Lagg^(1+eps_l^(-1)))^(1-eps_c^(-1))/(1-beta);

% Moments (va_share and io network)
va_share = mu.^(sigma_q^(-1)).*(Ydef./Qdef).^(1-sigma_q^(-1));
Pm_inverted = 1 ./ Pm; % Invert the elements of Pm
ratio = (P * Pm_inverted.').^(1 - sigma_m); % Compute the ratio P * Pm_inverted', and then raise it to the power (1 - sigma_m)
ionet = Gamma_M .* ratio; % Multiply Gamma_M element-wise by the ratio to obtain the new matrix

% Equilibrium
Qrc_loss = Q./Qrc - 1;
C_loss = P./MgUtCmod - 1;
L_loss = MgUtLmod ./MPLmod - 1;
K_loss = Pk./MPKmod - 1;
Pm_loss = Pm./Pmdef - 1;
M_loss = M./Mmod - 1;
Mout_loss = Mout./Moutmod - 1;
Pk_loss = Pk./Pkdef - 1;
Iout_loss = Iout./Ioutmod - 1;
Qdef_loss = Q./ Qdef - 1;
Ydef_loss = Y./ Ydef - 1;
Caggdef_loss = Cagg/Caggdef - 1;
Laggdef_loss = Lagg/Laggdef - 1;
%norm_loss = Cagg/1-1;
% Moments
mu_loss = va_share(1:n_sectors)./vashare_data(1:n_sectors) - 1;
Gamma_M_loss = ionet(1:n_sectors-1,:)./ionet_data(1:n_sectors-1,:)-1;
Gamma_M_loss = Gamma_M_loss(:);

fx = zeros(12*n_sectors+2+n_sectors*(n_sectors-1),1);
fx(1:n_sectors) = C_loss;
fx(n_sectors+1:2*n_sectors) = L_loss;
fx(2*n_sectors+1:3*n_sectors) = K_loss;
fx(3*n_sectors+1:4*n_sectors) = Pm_loss;
fx(4*n_sectors+1:5*n_sectors) = M_loss;
fx(5*n_sectors+1:6*n_sectors) = Mout_loss;
fx(6*n_sectors+1:7*n_sectors) = Pk_loss;
fx(7*n_sectors+1:8*n_sectors) = Iout_loss;
fx(8*n_sectors+1:9*n_sectors) = Qrc_loss;
fx(9*n_sectors+1:10*n_sectors) = Qdef_loss;
fx(10*n_sectors+1:11*n_sectors) = Ydef_loss;
fx(11*n_sectors+1) = Caggdef_loss;
fx(11*n_sectors+2) = Laggdef_loss;
%fx(11*n_sectors+3) = norm_loss;
fx(11*n_sectors+3:12*n_sectors+2) = mu_loss;
fx(12*n_sectors+3:12*n_sectors+2+n_sectors*(n_sectors-1)) = Gamma_M_loss;
% disp(mu_loss)
%% Print Section %%

if print
    % print steady state values
    disp(' ');

    disp('Analytic steady state');

    disp(' ');

    disp ('--- Prices ---')
    disp('Sectoral Price Indices (P_j): ');
    disp(P)
    disp('Capital Price Indices (P.^k_j): ');
    disp(Pk)
    disp('Intermediate Price Indices (P.^m_j): ');
    disp(Pm)

    disp('--- Quantities ---')
    disp('Sectoral Gross Output (Q_j): ')
    disp(Q)
    disp('Sectoral Value Added (Y_j): ')
    disp(Y)
    disp('Sectoral Intermediates (M_j): ')
    disp(M)
    disp('Sectoral Investment (I_j): ')
    disp(I)
    disp('Sectoral Labor (L_j): ')
    disp(L)
    disp('Sectoral Capital (K_j): ')
    disp(K)
    
    disp('--- Shares over nominal consumption  ---')
    disp('Sectoral Gross Output (P_jQ_j/PC): ')
    disp(P.*Q/(P'*C))
    disp('Sectoral Value Added (P_jY_j/PC): ')
    disp(P.*Y/(P'*C))
    disp('Sectoral Intermediates (Pm_jM_j/PC): ')
    disp(Pm.*M/(P'*C))
    disp('Sectoral Investment (Pk_jI_j/PC): ')
    disp(Pk.*I/(P'*C))
    disp('Sectoral Capital (Pk_jK_j/PC): ')
    disp(Pk.*K/(P'*C))
    
    disp('--- Aggregate Quantities ---')
    disp(['Aggregate Consumption (C): ',num2str(Cagg)])
    disp(['Aggregate Labor (L): ',num2str(Lagg)])
    
    disp('--- Aggregate Shares over nominal consumption  ---')
    disp('Sectoral Gross Output (PQ/PC): ')
    disp(P'*Q/(P'*C))
    disp('Aggregate Share Value Added (PY/PC): ')
    disp(P'*Y/(P'*C))
    disp('Aggregate Share of Intermediates (Pm M/PC): ')
    disp(Pm'*M/(P'*C))
    disp('Aggregate Share of Investment (Pk I/PC): ')
    disp(Pk'*I/(P'*C))
    disp('capital over Consumption (Pk K/PC): ')
    disp(Pk'*K/(P'*C))
    
    
    disp('--- Welfare ---')
    disp(V);

    disp('MgUt Cagg')
    disp(MgUtCagg);
    disp('Ldef')
    disp(Laggdef);
    disp('Pagg')
    disp(Pagg);
    disp('theta')
    disp(theta)
    

end

%% Return Output

StStval.parameters    = struct('parn_sectors', n_sectors, 'parbeta', beta, 'pareps_c', eps_c, 'pareps_l', eps_l, 'parphi', phi, 'partheta', theta, ...
    'parsigma_c', sigma_c, 'parsigma_m', sigma_m, 'parsigma_q', sigma_q, 'parsigma_y', sigma_y,'parsigma_I', sigma_I, 'parsigma_l', sigma_l, ...
    'paralpha', alpha, 'pardelta', delta, 'parmu', mu, 'parrho', rho, 'parxi', xi, ...
    'parGamma_I', Gamma_I, 'parGamma_M', Gamma_M, 'parSigma_A', Sigma_A);
StStval.policies_ss = [sol(1:11*n_sectors+2);log(P'*Y);log(Pk'*I);log(Pm'*M)];
StStval.endostates_ss = log(K);
StStval.Cagg_ss = Cagg;
StStval.Lagg_ss = Lagg;
StStval.Yagg_ss = P'*Y;
StStval.Iagg_ss = Pk'*I;
StStval.Magg_ss = Pm'*M;
StStval.V_ss = V;


end