%%% This code runs the dynare codes for Carvalho, Covarrubias and Nuño (2022)
clear; 
clearvars -global;    
clc; 

% Options (change here)
save_exper_ind = 1; % 1 to save data  and graphs for the experiment.
big_sector_ind = 0; % 1 to analyze big sectors, 0 to analize intermediate suppliers.

% date = datetime('today', 'Format', 'yyyy-MM-dd');
date = "Sep7";
exp_label = "_ghh_va_highcomp";

% Label creation
if big_sector_ind ==1
    focus_label = "bigsec";
    sector_indices = [24];
else
    focus_label = "intermsec";
    % sector_indices = [1, 5, 6];
    sector_indices = [1];
end
sector_labels = SectorLabel(sector_indices);
save_label = strcat(date, exp_label, focus_label);

% Length of IRs
N=60;
ax=0:N-1;
disp(save_label)

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

%%%%% Externally calibrated parameters %%%%%

% Depreciation rates
params.delta = mean(depratbea,2); % Average implied depreciation rate by industry 1947-2018 (BEA fixed assets)

% common values in literature
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.5;
params.theta = 1;
% COBB-DOUGLAS case
sigma_c = 0.99;
sigma_m = 0.99;
sigma_q = 0.99;
sigma_y = 0.99; 
sigma_I = 0.99;
sigma_l = 0.5; 

% useful parameteres for later
params.n_sectors = 37;
params.IRshock = 0.2231;
% params.IRshock = 0.3567;
% length of simulations
params.simul_T = 5000;

% process for TFP
load TFP_process;
params.rho = modrho;
% params.rho = 0.6*ones([37,1]);
params.Sigma_A = modvcv;

% Capital adjustment and labor reallocation
params.phi = 2; % we will recalibrate this later

% expenditure shares (for matching)
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

% Extracting the maximum client of each sectors and the ranking of clients
client_indices = zeros(numel(sector_indices),1);
ranking = zeros(numel(sector_indices),params.n_sectors);

for s_idx = sector_indices
    % we exclude the own sector when computing the client
    ionet_without_sector = [ionet_data(s_idx,1:s_idx-1),ionet_data(s_idx,s_idx+1:end)];
    [max_value, col_index] = max(ionet_without_sector);
    if col_index >= s_idx
        col_index = col_index+1;
    end
    client_indices(find(sector_indices == s_idx)) = col_index;
    shares_vector = ionet_data(s_idx,:);
    sorted_ionet = sort(shares_vector,'descend');
    [~, rank] = ismember(shares_vector , sorted_ionet);
    % artificially placing the sector as the 1st in the ranking when it's not
    if rank(s_idx) ~= 1
        rank(rank<rank(s_idx)) = rank(rank<rank(s_idx))+1;
        rank(s_idx) = 1;
    end
    ranking(find(sector_indices == s_idx),:) = rank;
end
client_labels = SectorLabel(client_indices);

disp('*** FINISHED STORING PARAMETERS IN params STRUCTURE ***');

%% Solve Steady State
% elasticities of substitution
params.sigma_c = 0.99;
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

load SS_CDsolution_norm_permanent.mat
% sol_guess=[zeros([11*params.n_sectors,1])-1;0;-1;1];
sol_guess = sol_init;

% Assuming intensitiy shares are equal to expenditure shares
total_tic = tic;
fh_compStSt         = @(x) ProdNetRbc_SS(x,params, 0);
[sol_init,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
[fx,ModData_CD]        = ProdNetRbc_SS(sol_init,params,1);
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run.\n', elapsed_time);

if save_exper_ind==1
    save SS_CDsolution_norm.mat sol_init
end

disp('*** FINISHED SOLVING STEADY STATE WITH SIGMAS EQUAL TO 1 ***');


%% change sigma_l

% Initial guess (from previous section or uncomment next line)
% load SS_CDsolution_normv2.mat
sol_guess = sol_init;

% loop over grid of values to get final answer
gridpoints = 5;
sigma_l_grid = linspace(0.99,sigma_l,gridpoints);
total_tic = tic;
for i = 1:gridpoints
    iteration_tic = tic;
    params.sigma_l = sigma_l_grid(i);
    fh_compStSt         = @(x) ProdNetRbc_SS(x,params, 0);
    [sol_init,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
    [fx,ModData_CD]        = ProdNetRbc_SS(sol_init,params,1);
    sol_guess = sol_init;
    disp(sprintf('For sigma_l = %0.2f : %s',params.sigma_l,ExitFlag(exfl)));
    iteration_time = toc(iteration_tic);
    fprintf('It took %.4f seconds to run iteration number %d \n', iteration_time,i);
end
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run this loop.\n', elapsed_time);
disp('*** FINISHED SOLVING STEADY STATE WITH LOW SIGMA L***');

%% match expenditure shares for value added production and IO network (as an intermediate step)

% Intial guess
% load SS_CDsolution_normv2.mat
mu_guess = vash_data;
Gamma_M_guess = ionet_data(1:params.n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
sol_guess = [sol_init;log(mu_guess);log(Gamma_M_guess)];

% loop over grid of values to get final answer
gridpoints = 5;
sigma_m_grid = linspace(0.99,sigma_m,gridpoints);
sigma_q_grid = linspace(0.99,sigma_q,gridpoints);
total_tic = tic;
for i = 1:gridpoints
    iteration_tic = tic;
    params.sigma_m = sigma_m_grid(i);
    params.sigma_q = sigma_q_grid(i);
    fh_compStSt         = @(x) ProdNetRbc_SS_vaioshares(x,params, 0);
    [sol_partial,fval,exfl]  = fsolve(fh_compStSt,sol_guess,options);
    [fx,ModData_CD]        = ProdNetRbc_SS_vaioshares(sol_partial,params,1);
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

% load SS_CDsolution_normv2.mat
xi_guess = conssh_data;
mu_guess = ModData_CD.parameters.parmu;
alpha_guess = capsh_data;
Gamma_M_guess = ModData_CD.parameters.parGamma_M(1:params.n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
Gamma_I_guess = invnet_data(1:params.n_sectors-1, :);
Gamma_I_guess = Gamma_I_guess(:);
sol_guess = [sol_partial(1:11*params.n_sectors+3);log(xi_guess);log(mu_guess);log(alpha_guess);log(Gamma_M_guess);log(Gamma_I_guess)];

% loop over grid of values to get final answer
gridpoints = 5;
sigma_c_grid = linspace(0.99,sigma_c,gridpoints);
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
    [fx,ModData_CD]        = ProdNetRbc_SS_expshares(sol_partial,params,1);
    sol_guess = sol_partial;
    disp(sprintf('For sigma_c = %0.2f, sigma_y = %0.2f and sigma_I = %0.2f : %s',params.sigma_c,params.sigma_y,params.sigma_I,ExitFlag(exfl)));
    iteration_time = toc(iteration_tic);
    fprintf('It took %.4f seconds to run iteration number %d \n', iteration_time,i);
end
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run this loop.\n', elapsed_time);

if save_exper_ind == 1
    sol_guess = sol_partial;
    filename_sol_guess = strcat('sol_guess_sq', num2str(params.sigma_q), '_sl', num2str(params.sigma_l), '_', save_label, '.mat');
    filename_params = strcat('params_', save_label, '.mat');
    save(filename_sol_guess, 'sol_guess');
    save(filename_params, 'params');
end

disp('*** FINISHED SOLVING STEADY STATE ***');
%% save results
% save SS_guesses/SS_allexpshare sol_final
% Save data to open in Python

if save_exper_ind == 1
    filename_Mod = strcat('output/RbcProdNet_ModData_CD', save_label, '.mat');
    save(filename_Mod, 'ModData_CD');
end
 
policies_ss = ModData_CD.policies_ss;
k_ss = ModData_CD.endostates_ss;

% Save a matlab structure to pass to dyanre. It needs to have each aprameter as a different variable

params_vars = struct2cell(ModData_CD.parameters);
params_names = fieldnames(ModData_CD.parameters);
for i = 1:numel(params_vars)
    assignin('base', params_names{i}, params_vars{i});
end

% Save data to open in Dynare (if you change name, change it in mod file and steady_state files as well)
save ModStruct_temp par* policies_ss k_ss N ax

disp('*** CREATED TEMP STRUCTURE ModStruct_temp TO PASS TO DYNARE ***');
disp('*** STORED THE MODEL INFO IN output/RbcProdNet_ModData_CD_Jan24. ***');

%% Solve Log-Linearized Model in Dynare
tic;
dynare stoch_simul;
elapsed_time = toc;
fprintf('It took %.4f seconds to run.\n', elapsed_time);
clearvars -except oo_ options_  M_ par* ModData_CD policies_ss k_ss sector_indices save_exper_ind sector_labels save_label N ax client_indices client_labels ranking
disp('*** FINISHED SOLVING DYNARE MODEL. SOLUTION IS STORED IN oo_ and M_ GLOBALS ***');

%% Simulating the model
tic;
modorder = 1; % order of the simulation
shockssim = mvnrnd(zeros([parn_sectors,1]),params.Sigma_A,params.simul_T); % 100000 random draws of the shock.
ss_values=oo_.steady_state; % get the steafy state 
k_ss = ss_values(1:parn_sectors);
policies_ss = ss_values(2*parn_sectors+1:13*parn_sectors+6);
display([ss_values(2*parn_sectors+1:13*parn_sectors+6),ModData_CD.policies_ss]);

shocks_sd = sqrt(var(shockssim,0,1)).';

% Simulating using dynare's function simult_
dynare_simul = simult_(M_,options_, oo_.steady_state,oo_.dr,shockssim,modorder);
varlev=exp(dynare_simul(1:13*parn_sectors+2,:)); % variables in levels
variables_var = diag(var(dynare_simul)); % variance

% Get aggregates of simulation
Cagg_ss = exp(ss_values(13*parn_sectors+1));
Lagg_ss = exp(ss_values(13*parn_sectors+2));
Yagg_ss = exp(ss_values(13*parn_sectors+3));
Iagg_ss = exp(ss_values(13*parn_sectors+4));
Magg_ss = exp(ss_values(13*parn_sectors+5));
V_ss = ss_values(13*parn_sectors+6);
A_Simul = varlev(parn_sectors+1:2*parn_sectors,1:params.simul_T);
Cagg_Simul = exp(dynare_simul(13*parn_sectors+1,1:params.simul_T));
Lagg_Simul = exp(dynare_simul(13*parn_sectors+2,1:params.simul_T));
V_Simul = dynare_simul(13*parn_sectors+6,1:params.simul_T);
Vc_Simul = dynare_simul(13*parn_sectors+7,1:params.simul_T);

% calculate simulation based variance:
states_sd = sqrt(variables_var(1:2*parn_sectors));
policies_sd = sqrt(variables_var(2*parn_sectors+1:13*parn_sectors+2));

% % store results
Simul = [A_Simul;Cagg_Simul;Lagg_Simul;Vc_Simul];
Simul_sd = [sqrt(var(A_Simul,0,2));sqrt(var(Cagg_Simul,0,2))./Cagg_ss;sqrt(var(Lagg_Simul,0,2))./Lagg_ss];

% get the matrices to recover the state space representation 
%S(t) = A*S(t-1) + B*e(t), evolution of the state
%X(t) = C*S(t-1) + D*e(t); evolution of the rest of the variables

ipred = M_.nstatic+(1:M_.nspred)'; %get state indices
[A,B] = kalman_transition_matrix(oo_.dr,ipred,1:M_.nspred,M_.exo_nbr); %get state transition matrices
obs_var=oo_.dr.inv_order_var(options_.varobs_id);
[C,D] = kalman_transition_matrix(oo_.dr,obs_var,1:M_.nspred,M_.exo_nbr);

% save resutls
if save_exper_ind==1
    save output/RbcProdNet_solution_CD_Mar23 ModData_CD states_sd policies_sd shocks_sd Cagg_ss Lagg_ss V_ss A B C D Simul Simul_sd
end
shockssim_data = shockssim';
elapsed_time = toc;
fprintf('It took %.4f seconds to run.\n', elapsed_time);
disp('*** FINISHED SIMULATION OF DYNARE POLICIES. ***');
disp('*** POLICIES AND SIMULATION STATISTICS SOLUTION IS STORED IN output/RbcProdNet_solution_CD_... ***');

%% Get Impulse Response
modorder = 1;
irs_loglin = cell(1, numel(sector_indices));
shocks_ir = cell(1, numel(sector_indices));
range_padding = 0.1;
tic;
% Loop over sectors
for idx = 1:numel(sector_indices)
    sector_idx = sector_indices(idx);
    sector_label = sector_labels{idx};
    client_idx = client_indices(idx);
    client_label = client_labels{idx};
    title_sector = strcat(sector_label, ' shock');

    % Create initial value with shock
    initval = oo_.steady_state;
    initval(parn_sectors+sector_idx)=-params.IRshock;
    shockssim = zeros([100, parn_sectors]); % we put the shock in initval not here
    dynare_simul = simult_(M_, options_, initval, oo_.dr, shockssim, modorder);

    % Calculate aggregate variables

    A_ir = exp(dynare_simul(parn_sectors + sector_idx, :));
    C_ir = dynare_simul(13 * parn_sectors + 1, :) - log(Cagg_ss);
    L_ir = dynare_simul(13 * parn_sectors + 2, :) - log(Lagg_ss);
    V_ir = dynare_simul(13 * parn_sectors + 6, :);
    Vc_ir = dynare_simul(13 * parn_sectors + 7, :);
    Y_ir = dynare_simul(13 * parn_sectors + 3, :) - initval(13 * parn_sectors + 3);
    I_ir = dynare_simul(13 * parn_sectors + 4, :) - initval(13 * parn_sectors + 4);
    M_ir = dynare_simul(13 * parn_sectors + 5, :) - initval(13 * parn_sectors + 5);
    
    % Calculate sectoral output variables
    Cj_ir = (dynare_simul(2 * parn_sectors + sector_idx, :) - policies_ss(0*parn_sectors+sector_idx));
    Pj_ir = (dynare_simul(10 * parn_sectors + sector_idx, :) - policies_ss(8*parn_sectors+sector_idx));
    Ioutj_ir = (dynare_simul(9 * parn_sectors + sector_idx, :) - policies_ss(7*parn_sectors+sector_idx));
    Moutj_ir = (dynare_simul(7 * parn_sectors + sector_idx, :) - policies_ss(5*parn_sectors+sector_idx));
    
    % Calculate sectoral input variables
    Lj_ir = (dynare_simul(3 * parn_sectors + sector_idx, :) - policies_ss(1*parn_sectors+sector_idx));
    Ij_ir = (dynare_simul(8 * parn_sectors + sector_idx, :) - policies_ss(6*parn_sectors+sector_idx));
    Mj_ir = (dynare_simul(6 * parn_sectors + sector_idx, :) - policies_ss(4*parn_sectors+sector_idx));
    Yj_ir = (dynare_simul(12 * parn_sectors + sector_idx, :) - policies_ss(10*parn_sectors+sector_idx));
    Qj_ir = (dynare_simul(11 * parn_sectors + sector_idx, :) - policies_ss(9*parn_sectors+sector_idx));
    % Kj_ir = exp(dynare_simul(sector_idx, :));
    Kj_ir = (dynare_simul(sector_idx, :) - k_ss(sector_idx));
    % Calculate client sectoral output variables

    A_client_ir = exp(dynare_simul(parn_sectors + client_idx, :));

    Cj_client_ir = (dynare_simul(2 * parn_sectors + client_idx, :) - policies_ss(0*parn_sectors+client_idx));
    Pj_client_ir = (dynare_simul(10 * parn_sectors + client_idx, :) - policies_ss(8*parn_sectors+client_idx));
    Ioutj_client_ir = (dynare_simul(9 * parn_sectors + client_idx, :) - policies_ss(7*parn_sectors+client_idx));
    Moutj_client_ir = (dynare_simul(7 * parn_sectors + client_idx, :) - policies_ss(5*parn_sectors+client_idx));
    
    % Calculate client sectoral input variables
    Lj_client_ir = (dynare_simul(3 * parn_sectors + client_idx, :) - policies_ss(1*parn_sectors+client_idx));
    Pmj_client_ir = (dynare_simul(5 * parn_sectors + client_idx, :) - policies_ss(3*parn_sectors+client_idx));
    Mj_client_ir = (dynare_simul(6 * parn_sectors + client_idx, :) - policies_ss(4*parn_sectors+client_idx));
    Ij_client_ir = (dynare_simul(8 * parn_sectors + client_idx, :) - policies_ss(6*parn_sectors+client_idx));
    Yj_client_ir = (dynare_simul(12 * parn_sectors + client_idx, :) - policies_ss(10*parn_sectors+client_idx));
    Qj_client_ir = (dynare_simul(11 * parn_sectors + client_idx, :) - policies_ss(9*parn_sectors+client_idx));
    
    % Calculate client expenditure share on affected input
    Pmj_lev_client_ir = exp(dynare_simul(5 * parn_sectors + client_idx, :));
    Pmj_lev_client_ss = exp(policies_ss(3 * parn_sectors + client_idx));
    Pj_lev_ir = exp(dynare_simul(10 * parn_sectors + sector_idx, :));
    Pj_lev_ss = exp(policies_ss(8 * parn_sectors + sector_idx));
    gammaij_lev_client_ir = params.Gamma_M(sector_idx,client_idx).*(Pj_lev_ir./Pmj_lev_client_ir).^(1-params.sigma_m);
    gammaij_lev_client_ss = params.Gamma_M(sector_idx,client_idx)*(Pj_lev_ss/Pmj_lev_client_ss)^(1-params.sigma_m);
    gammaij_client_ir = log(gammaij_lev_client_ir)-log(gammaij_lev_client_ss);

    % Store results 
    IRSLoglinCD{idx} = [A_ir; C_ir; L_ir; Vc_ir; Cj_ir; Pj_ir; Ioutj_ir; Moutj_ir; Lj_ir; Ij_ir; Mj_ir; Yj_ir; Qj_ir; ...
                        A_client_ir; Cj_client_ir; Pj_client_ir; Ioutj_client_ir; Moutj_client_ir; Lj_client_ir; Ij_client_ir; ...
                        Mj_client_ir; Yj_client_ir; Qj_client_ir; Kj_ir; Y_ir; Pmj_client_ir; gammaij_client_ir];
    % IRSLoglinCD = struct('A_ir', A_ir, 'C_ir', C_ir, 'L_ir', L_ir, 'Vc_ir', Vc_ir, 'Cj_ir', Cj_ir, 'Pj_ir', Pj_ir, ...
    %                     'Ioutj_ir', Ioutj_ir, 'Moutj_ir', Moutj_ir, 'Lj_ir', Lj_ir, 'Ij_ir', Ij_ir, 'Mj_ir', Mj_ir, ...
    %                     'Yj_ir', Yj_ir, 'Qj_ir', Qj_ir, 'A_client_ir', A_client_ir, 'Cj_client_ir', Cj_client_ir, ...
    %                     'Pj_client_ir', Pj_client_ir, 'Ioutj_client_ir', Ioutj_client_ir, 'Moutj_client_ir', Moutj_client_ir, ...
    %                     'Lj_client_ir', Lj_client_ir, 'Ij_client_ir', Ij_client_ir, 'Mj_client_ir', Mj_client_ir, ...
    %                     'Yj_client_ir', Yj_client_ir, 'Qj_client_ir', Qj_client_ir, 'Kj_ir', Kj_ir);
    % IRSLoglinCD = struct('A_ir_CD', A_ir, 'C_ir_CD', C_ir, 'L_ir_CD', L_ir, 'Vc_ir_CD', Vc_ir, 'Cj_ir_CD', Cj_ir, 'Pj_ir_CD', Pj_ir, ...
    %                     'Ioutj_ir_CD', Ioutj_ir, 'Moutj_ir_CD', Moutj_ir, 'Lj_ir_CD', Lj_ir, 'Ij_ir_CD', Ij_ir, 'Mj_ir_CD', Mj_ir, ...
    %                     'Yj_ir_CD', Yj_ir, 'Qj_ir_CD', Qj_ir, 'A_client_ir_CD', A_client_ir, 'Cj_client_ir_CD', Cj_client_ir, ...
    %                     'Pj_client_ir_CD', Pj_client_ir, 'Ioutj_client_ir_CD', Ioutj_client_ir, 'Moutj_client_ir_CD', Moutj_client_ir, ...
    %                     'Lj_client_ir_CD', Lj_client_ir, 'Ij_client_ir_CD', Ij_client_ir, 'Mj_client_ir_CD', Mj_client_ir, ...
    %                     'Yj_client_ir_CD', Yj_client_ir, 'Qj_client_ir_CD', Qj_client_ir, 'Kj_ir_CD', Kj_ir);

end
    % Save them!
    if save_exper_ind==1
        save('IRSLoglinCD.mat', 'IRSLoglinCD');
        disp(['Data saved to ', 'IRSLoglinCD.mat']); 
    end
