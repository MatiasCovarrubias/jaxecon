%%% This code runs the dynare codes for Carvalho, Covarrubias and Nu�o (2022)
clear; 
clearvars -global;    
clc; 

% Options (change here)
save_exper_ind = 1; % 1 to save data  and graphs for the experiment.

% date = datetime('today', 'Format', 'yyyy-MM-dd');
date = "Feb29_24";
exp_label = "_nonlinear_v4";
save_label = strcat(date, exp_label);

% optimization options
options  = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',10000000,'MaxIter',10000);

%% Calibration

%%%%% Calibration targets and Data based parameters %%%%%

load calibration_data.mat

dim=37;

% Consumption Expenditure Shares
conssh_data=mean(Cons47bea./repmat(sum(Cons47bea,2),1,dim))';

% Capital expenditure  shares
capshbea(capshbea<0.05)=0.05; % Account for some odd swings in capital shares that generate very low numbers
capsh_data=mean(capshbea,2); 

% Value added share of gross output
vash_data=(mean(VA47bea./GO47bea))'; %Average Value added/Gross Output Ratio

% Investment matrix
invnet_data = mean(invmat,3);
invnet_data(invnet_data < 0.001) = 0.001;
invnet_data = invnet_data ./ sum(invnet_data, 1);

% Input-Output Matrix
ionet_data=mean(IOmat47bea./repmat(sum(IOmat47bea),[dim 1 1]),3);
ionet_data(ionet_data < 0.001) = 0.001;
ionet_data = ionet_data ./ sum(ionet_data, 1);
% Secotral investment volatility (to do)

% Sectoral labor reallocation (to do)

%% Empirical Targets for Volatility Matching (Step 1)
% Compute empirical targets from the data
empiricalTargets = struct();

% Extract raw data (rows = years, cols = sectors)
InvRaw_temp = InvRaw; % Investment data (71 x 37)
EMP_raw_temp = EMP_raw; % Labor data (71 x 37)
VA_raw_temp = VA_raw; % Value added data (71 x 37)

% Handle zeros/missing values to avoid log(0) issues
epsilon = 1e-10;
InvRaw_temp(InvRaw_temp <= 0) = epsilon;
EMP_raw_temp(EMP_raw_temp <= 0) = epsilon;
VA_raw_temp(VA_raw_temp <= 0) = epsilon;

% Build sector weights from value added (time-averaged, normalized)
w_emp = mean(VA_raw_temp, 1); % Average VA across time for each sector
weights_emp = w_emp ./ sum(w_emp); % Normalize to sum to 1

% Transform to log-differences in percent (rows=time, cols=sectors)
dlogI = 100 * diff(log(InvRaw_temp), 1, 1); % log-diff of investment
dlogL = 100 * diff(log(EMP_raw_temp), 1, 1); % log-diff of labor

% Compute sectoral volatilities (std over time for each sector)
sigmaIi = std(dlogI, 0, 1); % 1 x 37 vector
sigmaLi = std(dlogL, 0, 1); % 1 x 37 vector

% Aggregate with value-added weights
sigmaI_emp = sum(weights_emp .* sigmaIi);
sigmaL_emp = sum(weights_emp .* sigmaLi);

% Store in struct
empiricalTargets.sigmaI = sigmaI_emp;
empiricalTargets.sigmaL = sigmaL_emp;
empiricalTargets.weights = weights_emp;
empiricalTargets.sigmaIi = sigmaIi;
empiricalTargets.sigmaLi = sigmaLi;

fprintf('\n*** EMPIRICAL TARGETS ***\n');
fprintf('Value-added weighted investment volatility: %.4f\n', sigmaI_emp);
fprintf('Value-added weighted labor volatility: %.4f\n', sigmaL_emp);
fprintf('***************************\n\n');

%%%%% Externally calibrated parameters %%%%%

% Depreciation rates
params.delta = mean(depratbea,2); % Average implied depreciation rate by industry 1947-2018 (BEA fixed assets)

% common values in literature
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.5;
params.theta = 1;

% desired elasticied of substitution (we will work towards them in params
% struct later)
sigma_c = 0.5;
sigma_m = 0.01;
sigma_q = 0.2; % we will recalibrate this later
sigma_y = 0.6; 
sigma_I = 0.5;
sigma_l = 0.5; % we will recalibrate this later

% useful parameteres for later
params.n_sectors = 37;
params.IRshock = 0.4;
% length of simulations
params.simul_T = 100;

% process for TFP
load TFP_process_nonlinear;
params.rho = modrho;
% params.rho = 0.7*ones([37,1]);
params.Sigma_A = modvcv;

% Capital adjustment and labor reallocation
params.phi = 4; % we will recalibrate this later

% expenditure shares (for matching)
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

disp('*** FINISHED STORING PARAMETERS IN params STRUCTURE ***');

%% Solve Steady State under Cobb Douglas and high sigma_l
% elasticities of substitution
params.sigma_c = 0.5;
params.sigma_m = 0.99;
params.sigma_q = 0.99;
params.sigma_y = 0.99;
params.sigma_I = 0.99;
params.sigma_l = 0.99;
% Intesity shares in CD case
params.xi = conssh_data;
params.alpha = capsh_data;
params.mu = vash_data;
params.Gamma_M = ionet_data;
params.Gamma_I = invnet_data;

load SS_CDsolution_norm.mat
% sol_guess=[zeros([11*params.n_sectors,1])-1;0;-1;1];
sol_guess = sol_init;

% Assuming intensitiy shares are equal to expenditure shares
total_tic = tic;
fh_compStSt         = @(x) ProdNetRbc_SS(x,params, 0);
[sol_init,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
[fx,ModData]        = ProdNetRbc_SS(sol_init,params,0);
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run.\n', elapsed_time);

if save_exper_ind==1
    save SS_CDsolution_norm.mat sol_init
end

disp('*** FINISHED SOLVING STEADY STATE WITH SIGMAS EQUAL TO 1 ***');


%% lower sigma_l

% Initial guess (from previous section or uncomment next line)
sol_guess = sol_init;

% loop over grid of values to get final answer
gridpoints = 8;
sigma_l_grid = linspace(0.9,sigma_l,gridpoints);
total_tic = tic;
for i = 1:gridpoints
    iteration_tic = tic;
    params.sigma_l = sigma_l_grid(i);
    fh_compStSt         = @(x) ProdNetRbc_SS(x,params, 0);
    [sol_init,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
    [fx,ModData]        = ProdNetRbc_SS(sol_init,params,0);
    sol_guess = sol_init;
    disp(sprintf('For sigma_l = %0.2f : %s',params.sigma_l,ExitFlag(exfl)));
    iteration_time = toc(iteration_tic);
    fprintf('It took %.4f seconds to run iteration number %d \n', iteration_time,i);
end
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run this loop.\n', elapsed_time);
disp('*** FINISHED SOLVING STEADY STATE WITH LOW SIGMA_L***');

%% match expenditure shares for value added production and IO network (as an intermediate step)

% Intial guess
mu_guess = vash_data;
Gamma_M_guess = ionet_data(1:params.n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
sol_guess = [sol_init;log(mu_guess);log(Gamma_M_guess)];

% loop over grid of values to get final answer
gridpoints = 8;
sigma_m_grid = linspace(0.99,sigma_m,gridpoints);
sigma_q_grid = linspace(0.99,sigma_q,gridpoints);
total_tic = tic;
for i = 1:gridpoints
    iteration_tic = tic;
    params.sigma_m = sigma_m_grid(i);
    params.sigma_q = sigma_q_grid(i);
    fh_compStSt         = @(x) ProdNetRbc_SS_vaioshares(x,params, 0);
    [sol_partial,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
    [fx,ModData]        = ProdNetRbc_SS_vaioshares(sol_partial,params,0);
    sol_guess = sol_partial;
    disp(sprintf('For sigma_m = %0.2f and sigma_q = %0.2f : %s',params.sigma_m,params.sigma_q,ExitFlag(exfl)));
    iteration_time = toc(iteration_tic);
    fprintf('It took %.4f seconds to run iteration number %d \n', iteration_time,i);
end
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run this loop.\n', elapsed_time);
disp('*** FINISHED SOLVING STEADY STATE WITH MATCHED VALUE ADDED AND IO SHARES ***');
%% match all expenditure shares

% Intial guess
xi_guess = conssh_data;
mu_guess = ModData.parameters.parmu;
alpha_guess = capsh_data;
Gamma_M_guess = ModData.parameters.parGamma_M(1:params.n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
Gamma_I_guess = invnet_data(1:params.n_sectors-1, :);
Gamma_I_guess = Gamma_I_guess(:);
sol_guess = [sol_partial(1:11*params.n_sectors+3);log(xi_guess);log(mu_guess);log(alpha_guess);log(Gamma_M_guess);log(Gamma_I_guess)];

% loop over grid of values to get final answer
gridpoints = 8;
sigma_c_grid = linspace(0.5,sigma_c,gridpoints);
sigma_y_grid = linspace(0.99,sigma_y,gridpoints);
sigma_I_grid = linspace(0.99,sigma_I,gridpoints);

total_tic = tic;
for i = 1:gridpoints
    iteration_tic = tic;
    params.sigma_c = sigma_c_grid(i);
    params.sigma_y = sigma_y_grid(i);
    params.sigma_I = sigma_I_grid(i);
    fh_compStSt         = @(x) ProdNetRbc_SS_expshares(x,params, 0);
    [sol_partial,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
    [fx,ModData]        = ProdNetRbc_SS_expshares(sol_partial,params,0);
    sol_guess = sol_partial;
    disp(sprintf('For sigma_c = %0.2f, sigma_y = %0.2f and sigma_I = %0.2f : %s',params.sigma_c,params.sigma_y,params.sigma_I,ExitFlag(exfl)));
    iteration_time = toc(iteration_tic);
    fprintf('It took %.4f seconds to run iteration number %d \n', iteration_time,i);
end
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run this loop.\n', elapsed_time);

if save_exper_ind == 1
    sol_guess = sol_partial;
    filename_sol_guess = strcat('output/','sol_guess_sq', num2str(params.sigma_q), '_sl', num2str(params.sigma_l), '_', save_label, '.mat');
    filename_params = strcat('output/','params_', save_label, '.mat');
    save(filename_sol_guess, 'sol_guess');
    save(filename_params, 'params');
end

disp('*** FINISHED SOLVING STEADY STATE ***');
%% save results

% Save data to open in Python
if save_exper_ind == 1
    filename_Mod = strcat('output/RbcProdNet_ModData_', save_label, '.mat');
    save(filename_Mod, 'ModData');
end
 
% Save a matlab structure to pass to dyanre. It needs to have each aprameter as a different variable

policies_ss = ModData.policies_ss;
k_ss = ModData.endostates_ss;

params_vars = struct2cell(ModData.parameters);
params_names = fieldnames(ModData.parameters);
for i = 1:numel(params_vars)
    assignin('base', params_names{i}, params_vars{i});
end

% Save data to open in Dynare (if you change name, change it in mod file and steady_state files as well)
save ModStruct_temp par* policies_ss k_ss

disp('*** CREATED TEMP STRUCTURE ModStruct_temp TO PASS TO DYNARE ***');
disp('*** STORED THE MODEL INFO IN output/RbcProdNet_ModData_Jan24. ***');

%% Solve Log-Linearized Model in Dynare
tic;
dynare stoch_simul;
elapsed_time = toc;
fprintf('It took %.4f seconds to run.\n', elapsed_time);
clearvars -except oo_ options_  M_ par* ModData policies_ss endostates_ss sector_indices save_exper_ind sector_labels save_label N ax client_indices client_labels ranking labels Cagg_ss Lagg_ss ar1resid*
disp('*** FINISHED SOLVING DYNARE MODEL. SOLUTION IS STORED IN oo_ and M_ GLOBALS ***'); 

%% Simulating the model
tic;
dim=parn_sectors;
% params.simul_T = 200;
modorder = 1; % order of the simulation
shockssim = mvnrnd(zeros([parn_sectors,1]),params.Sigma_A,params.simul_T); % 100000 random draws of the shock.
% shockssim = zeros([params.simul_T,37]);
% shockssim(1:70,:) = ar1resid; 
% shockssim = ar1resid; % 100000 random draws of the shock.

ss_values=oo_.steady_state; % get the steafy state 
k_ss = ss_values(1:parn_sectors);
policies_ss = ss_values(2*parn_sectors+1:13*parn_sectors+5);
% policies_ss(6*parn_sectors+1:7*parn_sectors) = k_ss; %we use as policy in python
display([ss_values(2*parn_sectors+1:13*parn_sectors+6),ModData.policies_ss]);

% Simulating using dynare's function simult_
dynare_simul = simult_(M_,options_, oo_.steady_state,oo_.dr,shockssim,modorder);
varlev=exp(dynare_simul(1:13*parn_sectors+5,:)); % variables in levels
variables_var = var(dynare_simul,0,2); % variance

% calculate simulation based variance:
shocks_sd = sqrt(var(shockssim,0,1)).';
states_sd = sqrt(variables_var(1:2*parn_sectors));
policies_sd = sqrt(variables_var(2*parn_sectors+1:13*parn_sectors+5));
% policies_sd(8*parn_sectors+1:9*parn_sectors) = states_sd(parn_sectors+1:2*parn_sectors);


% Simulating using the State Space Representation
[tn,sn] = size(oo_.dr.ghx); % dimensions, tn is total number of variables, sn is number of states

% get the matrices to recover the state space representation 
%S(t) = A*S(t-1) + B*e(t), evolution of the state
%X(t) = C*S(t-1) + D*e(t); evolution of the rest of the variables

% recover position of each variable in state space representation
k_ind = [1,dim];
a_ind = [dim+1,2*dim];
c_ind = [2*dim+1,3*dim];
l_ind = [3*dim+1,4*dim];
pk_ind = [4*dim+1,5*dim];
pm_ind = [5*dim+1,6*dim];
m_ind = [6*dim+1,7*dim];
mout_ind = [7*dim+1,8*dim];
i_ind = [8*dim+1,9*dim];
iout_ind = [9*dim+1,10*dim];
p_ind = [10*dim+1,11*dim];
q_ind = [11*dim+1,12*dim];
y_ind = [12*dim+1,13*dim];
cagg_ind = 13*dim+1;
lagg_ind = 13*dim+2;
yagg_ind = 13*dim+3;
iagg_ind = 13*dim+4;
magg_ind = 13*dim+5;

k_ind_inv = [oo_.dr.inv_order_var(k_ind(1)),oo_.dr.inv_order_var(k_ind(2))];
a_ind_inv = [oo_.dr.inv_order_var(a_ind(1)),oo_.dr.inv_order_var(a_ind(2))];

c_ind_inv = [oo_.dr.inv_order_var(c_ind(1)),oo_.dr.inv_order_var(c_ind(2))];
l_ind_inv = [oo_.dr.inv_order_var(l_ind(1)),oo_.dr.inv_order_var(l_ind(2))];
pk_ind_inv = [oo_.dr.inv_order_var(pk_ind(1)),oo_.dr.inv_order_var(pk_ind(2))];
pm_ind_inv = [oo_.dr.inv_order_var(pm_ind(1)),oo_.dr.inv_order_var(pm_ind(2))];
m_ind_inv = [oo_.dr.inv_order_var(m_ind(1)),oo_.dr.inv_order_var(m_ind(2))];
mout_ind_inv = [oo_.dr.inv_order_var(mout_ind(1)),oo_.dr.inv_order_var(mout_ind(2))];
i_ind_inv = [oo_.dr.inv_order_var(i_ind(1)),oo_.dr.inv_order_var(i_ind(2))];
iout_ind_inv = [oo_.dr.inv_order_var(iout_ind(1)),oo_.dr.inv_order_var(iout_ind(2))];
p_ind_inv = [oo_.dr.inv_order_var(p_ind(1)),oo_.dr.inv_order_var(p_ind(2))];
q_ind_inv = [oo_.dr.inv_order_var(q_ind(1)),oo_.dr.inv_order_var(q_ind(2))];
y_ind_inv = [oo_.dr.inv_order_var(y_ind(1)),oo_.dr.inv_order_var(y_ind(2))];
cagg_ind_inv = oo_.dr.inv_order_var(cagg_ind);
lagg_ind_inv = oo_.dr.inv_order_var(lagg_ind);
yagg_ind_inv = oo_.dr.inv_order_var(yagg_ind);
iagg_ind_inv = oo_.dr.inv_order_var(iagg_ind);
magg_ind_inv = oo_.dr.inv_order_var(magg_ind);


A=[oo_.dr.ghx(k_ind_inv(1):k_ind_inv(2),:); 
   oo_.dr.ghx(a_ind_inv(1):a_ind_inv(2),:)];
B=[oo_.dr.ghu(k_ind_inv(1):k_ind_inv(2),:); 
   oo_.dr.ghu(a_ind_inv(1):a_ind_inv(2),:)];

% We can check that A and B are correct
%get state indices
ipred = M_.nstatic+(1:M_.nspred)';
%get state transition matrices
[A_2,B_2] = kalman_transition_matrix(oo_.dr,ipred,1:M_.nspred,M_.exo_nbr);
disp('Is manually calc. A equal to dynare cal. A?');
disp(isequal(A,A_2));
disp('Is manually calc. B equal to dynare cal. B?');
disp(isequal(B,B_2));

% We construct policy parameters
C =[oo_.dr.ghx(c_ind_inv(1):c_ind_inv(2),:);
         oo_.dr.ghx(l_ind_inv(1):l_ind_inv(2),:);
         oo_.dr.ghx(pk_ind_inv(1):pk_ind_inv(2),:);
         oo_.dr.ghx(pm_ind_inv(1):pm_ind_inv(2),:);
         oo_.dr.ghx(m_ind_inv(1):m_ind_inv(2),:);
         oo_.dr.ghx(mout_ind_inv(1):mout_ind_inv(2),:);
         oo_.dr.ghx(i_ind_inv(1):i_ind_inv(2),:);
         oo_.dr.ghx(iout_ind_inv(1):iout_ind_inv(2),:);
         oo_.dr.ghx(p_ind_inv(1):p_ind_inv(2),:);
         oo_.dr.ghx(q_ind_inv(1):q_ind_inv(2),:);
         oo_.dr.ghx(y_ind_inv(1):y_ind_inv(2),:);
         oo_.dr.ghx(cagg_ind_inv,:);
         oo_.dr.ghx(lagg_ind_inv,:);
         oo_.dr.ghx(yagg_ind_inv,:);
         oo_.dr.ghx(iagg_ind_inv,:);
         oo_.dr.ghx(magg_ind_inv,:)];
D =[oo_.dr.ghu(c_ind_inv(1):c_ind_inv(2),:);
         oo_.dr.ghu(l_ind_inv(1):l_ind_inv(2),:);
         oo_.dr.ghu(pk_ind_inv(1):pk_ind_inv(2),:);
         oo_.dr.ghu(pm_ind_inv(1):pm_ind_inv(2),:);
         oo_.dr.ghu(m_ind_inv(1):m_ind_inv(2),:);
         oo_.dr.ghu(mout_ind_inv(1):mout_ind_inv(2),:);
         oo_.dr.ghu(i_ind_inv(1):i_ind_inv(2),:);
         oo_.dr.ghu(iout_ind_inv(1):iout_ind_inv(2),:);
         oo_.dr.ghu(p_ind_inv(1):p_ind_inv(2),:);
         oo_.dr.ghu(q_ind_inv(1):q_ind_inv(2),:);
         oo_.dr.ghu(y_ind_inv(1):y_ind_inv(2),:);
         oo_.dr.ghu(cagg_ind_inv,:);
         oo_.dr.ghu(lagg_ind_inv,:);
         oo_.dr.ghu(yagg_ind_inv,:);
         oo_.dr.ghu(iagg_ind_inv,:);
         oo_.dr.ghu(magg_ind_inv,:)];


% % store results
% Simul = [A_Simul;Cagg_Simul;Lagg_Simul;Vc_Simul];
% Simul_sd = [sqrt(var(A_Simul,0,2));sqrt(var(Cagg_Simul,0,2))./Cagg_ss;sqrt(var(Lagg_Simul,0,2))./Lagg_ss];

% save resutls
SolData = struct;
SolData.parameters = ModData.parameters;
SolData.k_ss = k_ss;
SolData.policies_ss = policies_ss;
SolData.states_sd = states_sd;
SolData.policies_sd = policies_sd;
SolData.shocks_sd = shocks_sd;
SolData.A = A;
SolData.B = B;
SolData.C = C;
SolData.D = D;


if save_exper_ind==1
    filename_Sol = strcat('output/RbcProdNet_SolData_', save_label, '.mat');
    save(filename_Sol, 'SolData');
%     save filename_sol ModData states_sd policies_sd shocks_sd V_ss A B C D Simul Simul_sd
end
shockssim_data = shockssim';
elapsed_time = toc;
fprintf('It took %.4f seconds to run.\n', elapsed_time);
disp('*** FINISHED SIMULATION OF DYNARE POLICIES. ***');
disp('*** POLICIES AND SIMULATION STATISTICS SOLUTION IS STORED IN output/RbcProdNet_SolData_`savelabel� ***');

%% Model Moments Computation (Step 2)
% Extract simulated sectoral series from dynare simulation
% Reconstruct variables in levels from log-deviations
I_sim = exp(dynare_simul(i_ind(1):i_ind(2),:))'; % Investment (T_sim x 37)
L_sim = exp(dynare_simul(l_ind(1):l_ind(2),:))'; % Labor (T_sim x 37)
Y_sim = exp(dynare_simul(y_ind(1):y_ind(2),:))'; % Output/Value Added proxy (T_sim x 37)

% Handle any potential numerical issues
epsilon = 1e-10;
I_sim(I_sim <= 0) = epsilon;
L_sim(L_sim <= 0) = epsilon;
Y_sim(Y_sim <= 0) = epsilon;

% Compute model volatilities with the same transformation as empirical
dlogI_sim = 100 * diff(log(I_sim), 1, 1); % log-diff of investment
dlogL_sim = 100 * diff(log(L_sim), 1, 1); % log-diff of labor

% Compute sectoral volatilities (std over time for each sector)
sigmaIi_model = std(dlogI_sim, 0, 1); % 1 x 37 vector
sigmaLi_model = std(dlogL_sim, 0, 1); % 1 x 37 vector

% Aggregate with time-averaged value-added weights from simulation
weights_model = mean(Y_sim, 1); % Average Y across time for each sector
weights_model = weights_model ./ sum(weights_model); % Normalize to sum to 1

% Compute weighted averages
sigmaI_model = sum(weights_model .* sigmaIi_model);
sigmaL_model = sum(weights_model .* sigmaLi_model);

% Compute moment errors and distance
eI = sigmaI_model - empiricalTargets.sigmaI;
eL = sigmaL_model - empiricalTargets.sigmaL;
distance = eI^2 + eL^2;

% Store results
modelMoments = struct();
modelMoments.sigmaI = sigmaI_model;
modelMoments.sigmaL = sigmaL_model;
modelMoments.weights = weights_model;
modelMoments.sigmaIi = sigmaIi_model;
modelMoments.sigmaLi = sigmaLi_model;

% Create results structure
volatMatchResults = struct();
volatMatchResults.empirical = empiricalTargets;
volatMatchResults.model = modelMoments;
volatMatchResults.errors = struct('eI', eI, 'eL', eL);
volatMatchResults.distance = distance;
volatMatchResults.params = struct('sigma_l', params.sigma_l, 'phi', params.phi);

% Display results
fprintf('\n*** MODEL VS EMPIRICAL MOMENTS ***\n');
fprintf('                     Empirical    Model      Error\n');
fprintf('Investment vol:      %.4f      %.4f    %.4f\n', empiricalTargets.sigmaI, sigmaI_model, eI);
fprintf('Labor vol:           %.4f      %.4f    %.4f\n', empiricalTargets.sigmaL, sigmaL_model, eL);
fprintf('Distance (sum of squared errors): %.6f\n', distance);
fprintf('Current parameters: sigma_l = %.4f, phi = %.4f\n', params.sigma_l, params.phi);
fprintf('***********************************\n\n');


%% Deterministic simulation
shockssim_determ = shockssim; %100000 random draws of the shock.
simul_T = params.simul_T ;
save ModStruct_temp par* policies_ss k_ss simul_T shockssim_determ;

dynare determ_simul         
dynare_simul= Simulated_time_series.data';
A_Simul_determ = exp(dynare_simul(1:parn_sectors,:));
C_Simul_determ = exp(dynare_simul(13*parn_sectors+1,:));
L_Simul_determ = exp(dynare_simul(13*parn_sectors+2,:));
V_Simul_determ = dynare_simul(13*parn_sectors+6,:);
Vc_Simul_determ = dynare_simul(13*parn_sectors+7,:);

Simul_determ = [A_Simul_determ; C_Simul_determ; L_Simul_determ; Vc_Simul_determ];
Simul_determ_sd = [sqrt(var(A_Simul_determ,0,2));sqrt(var(C_Simul_determ,0,2))./Cagg_ss;sqrt(var(L_Simul_determ,0,2))./Lagg_ss];

shockssim_determ_data = shockssim_determ';

Vc_tplus1 = mean(Simul(40,2:40));
Vc_tplus1_determ = mean(Simul_determ(40,2:40));
sigma_C = Simul_sd(38);
sigma_C_determ = Simul_determ_sd(38);
sigma_L = Simul_sd(39);
sigma_L_determ = Simul_determ_sd(39);

display([Vc_tplus1, Vc_tplus1_determ,sigma_C,sigma_C_determ,sigma_L, sigma_L_determ]); 
save output/RbcProdNet_solution_Mar25 Simul Simul_sd states_sd shocks_sd policies_sd A B C D irs_loglin irs_determ irs_irrev Simul_determ Simul_determ_sd
% clearvars -except oo_ options_  M_ par* states_sd K_determ_sd shockssim_determ shocks_sd policies_sd policies_determ_sd Simul_determ Simul_determ_sd irMin irRE irCons irMin_determ irRE_determ irCons_determ
disp('*** FINISHED RUNING DETERMINIST SIMULATION DYNRARE SCRIPTS ***')

