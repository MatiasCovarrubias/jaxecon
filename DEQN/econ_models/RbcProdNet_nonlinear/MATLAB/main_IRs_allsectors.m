%%% This code runs the dynare codes for Carvalho, Covarrubias and Nu�o (2022)
clear; 
clearvars -global;    
clc; 

% Options (change here)
save_exper_ind = 1; % 1 to save data  and graphs for the experiment.
big_sector_ind = 1; % 1 to analyze big sectors, 0 to analize intermediate suppliers.

% Labels
date = "_Oct_25";
exp_label = "nonlinear_pos_5";

% MODIFIED: We'll process all 37 sectors instead of specific ones
% choosing sectors to analyze
all_sectors = 1:37; % Loop through all 37 sectors
sector_indices = all_sectors; % Define sector_indices for backward compatibility

sector_labels = SectorLabel(all_sectors);
save_label = strcat(date, exp_label);

% Length of IRs
N=60;
ax=0:N-1;
disp(save_label)

% optimization options
options  = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',10000000,'MaxIter',10000);

%% Calibration

%%%%% Calibration targets and Data based parameters %%%%%

load calibration_data.mat
obs_stochss = readmatrix("obs_stochss.csv");
policies_stochss = readmatrix("policies_stochss.csv");

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
invnet_data(invnet_data < 0.01) = 0.01;
invnet_data = invnet_data ./ sum(invnet_data, 1);

% Input-Output Matrix
ionet_data=mean(IOmat47bea./repmat(sum(IOmat47bea),[dim 1 1]),3);
ionet_data(ionet_data < 0.01) = 0.01;
ionet_data = ionet_data ./ sum(ionet_data, 1);

% Secotral investment volatility (to do)

% Sectoral labor reallocation (to do)

%%%%% Externally calibrated parameters %%%%%

% Depreciation rates
params.delta = mean(depratbea,2); % Average implied depreciation rate by industry 1947-2018 (BEA fixed assets)

% common values in literature
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.33;
params.theta = 1;

% desired elasticied of substitution (we will work towards them in params
% struct later)
sigma_c = 0.5;
sigma_m = 0.01;
sigma_q = 0.5; % we will recalibrate this later
sigma_y = 0.8; 
sigma_I = 0.5;
sigma_l = 0.1; % we will recalibrate this later

% useful parameteres for later
params.n_sectors = 37;
%params.IRshock = -0.2231;
params.IRshock = -0.0513;
% params.IRshock = 0.3567;
% length of simulations
params.simul_T = 200;

% process for TFPh
load TFP_process;
params.rho = modrho;
% params.rho = 0.6*ones([37,1]);
params.Sigma_A = modvcv;

% Capital adjustment and labor reallocation
params.phi = 4; % we will recalibrate this later

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
labels = {sector_indices, sector_labels, client_indices, client_labels};
disp('*** FINISHED STORING PARAMETERS IN params STRUCTURE ***');

%% Solve Steady State under Cobb Douglas and high sigma_l
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
[fx,ModData]        = ProdNetRbc_SS(sol_init,params,1);
elapsed_time = toc(total_tic);
fprintf('It took %.4f seconds to run.\n', elapsed_time);

if save_exper_ind==1
    save SS_CDsolution_norm.mat sol_init
end

disp('*** FINISHED SOLVING STEADY STATE WITH SIGMAS EQUAL TO 1 ***');
disp(ModData.parameters.partheta)

%% lower sigma_l

% Initial guess (from previous section or uncomment next line)
sol_guess = sol_init;

% loop over grid of values to get final answer
gridpoints = 5;
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
disp(ModData.parameters.partheta)
%% match expenditure shares for value added production and IO network (as an intermediate step)

% Intial guess
mu_guess = vash_data;
Gamma_M_guess = ionet_data(1:params.n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
sol_guess = [sol_init;log(mu_guess);log(Gamma_M_guess)];

% loop over grid of values to get final answer
gridpoints = 10;
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
disp(ModData.parameters.partheta)
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
gridpoints = 30;
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
disp(ModData.parameters.partheta)
%% save results

% Save data to open in Python
if save_exper_ind == 1
    filename_Mod = strcat('output/RbcProdNet_ModData_', save_label, '.mat');
    save(filename_Mod, 'ModData');
end
 
% Save a matlab structure to pass to dyanre. It needs to have each aprameter as a different variable

policies_ss = ModData.policies_ss;
k_ss = ModData.endostates_ss;
Cagg_ss = exp(policies_ss(11*params.n_sectors+1));
Lagg_ss = exp(policies_ss(11*params.n_sectors+2));
params_vars = struct2cell(ModData.parameters);
params_names = fieldnames(ModData.parameters);
for i = 1:numel(params_vars)
    assignin('base', params_names{i}, params_vars{i});
end

% Save data to open in Dynare (if you change name, change it in mod file and steady_state files as well)
save ModStruct_temp par* policies_ss obs_stochss policies_stochss k_ss N ax

disp('*** CREATED TEMP STRUCTURE ModStruct_temp TO PASS TO DYNARE ***');
disp('*** STORED THE MODEL INFO IN output/RbcProdNet_ModData_date. ***');

%% Solve Log-Linearized Model in Dynare
tic;
dynare stoch_simul;
elapsed_time = toc;
fprintf('It took %.4f seconds to run.\n', elapsed_time);
clearvars -except oo_ options_  M_ par* ModData policies_ss k_ss sector_indices save_exper_ind sector_labels save_label N ax client_indices client_labels ranking labels Cagg_ss Lagg_ss
disp('*** FINISHED SOLVING DYNARE MODEL. SOLUTION IS STORED IN oo_ and M_ GLOBALS ***'); 

ss_values=oo_.steady_state; % get the steady state

% Store the loglin model and variables
M_loglin = M_;
oo_loglin = oo_;
options_loglin = options_;

%% Get Impulse Response
modorder = 1;
range_padding = 0.1;
original_steady_state = oo_loglin.steady_state; % Use the stored loglin steady state

% Create a struct to store all IRs
AllIRS = struct();

% Initialize arrays to store peak values, periods, and half-lives for all sectors
peak_values_shock = zeros(params.n_sectors, 1);
peak_values_loglin = zeros(params.n_sectors, 1);
peak_values_determ = zeros(params.n_sectors, 1);
amplifications = zeros(params.n_sectors, 1);  % New array for amplifications

peak_periods_shock = zeros(params.n_sectors, 1);
peak_periods_loglin = zeros(params.n_sectors, 1);
peak_periods_determ = zeros(params.n_sectors, 1);

half_lives_shock = zeros(params.n_sectors, 1);
half_lives_loglin = zeros(params.n_sectors, 1);
half_lives_determ = zeros(params.n_sectors, 1);

% Loop over all sectors (1 to 37)
for sector_idx = 1:params.n_sectors
    disp(['Processing sector ' num2str(sector_idx) ' of 37']);
    
    % ---- LOGLINEAR IR ----
    % Create a fresh copy of the loglin steady state for each sector
    steady_state = original_steady_state;
    steady_state(parn_sectors+sector_idx) = -params.IRshock;
    
    % Use the stored loglin model to simulate
    shockssim_ir = zeros([params.simul_T, parn_sectors]);
    dynare_simul = simult_(M_loglin, options_loglin, steady_state, oo_loglin.dr, shockssim_ir, modorder);
    IRSLoglin = ProcessIRs(dynare_simul, sector_idx, params, steady_state, labels, parn_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    
    % ---- DETERMINISTIC IR ----
   % DETERMINISTIC IR
    shocksim_0 = zeros([parn_sectors,1]);
    shocksim_0(sector_idx,1)=-params.IRshock;
    shockssim_ir = zeros([params.simul_T, parn_sectors]); % we put the shock in int vector, not here
    save ModStruct_temp par* policies_ss k_ss shockssim_ir shocksim_0;
    dynare determ_irs;
    dynare_simul = Simulated_time_series.data';
    IRSDeterm = ProcessIRs(dynare_simul,sector_idx, params,steady_state,labels,parn_sectors,k_ss, Cagg_ss, Lagg_ss,policies_ss);
    
    % Store IRs for this sector
    AllIRS.(['Sector_' num2str(sector_idx)]).IRSLoglin = IRSLoglin;
    AllIRS.(['Sector_' num2str(sector_idx)]).IRSDeterm = IRSDeterm;
    
    % Calculate peaks and half-lives
    C_loglin = IRSLoglin(2,1:100);
    C_determ = IRSDeterm(2,1:100);
    shock = 1 - IRSLoglin(1,1:100);
    
    [peak_value_shock, peak_period_shock, half_life_shock] = calculatePeaksAndHalfLives(shock);
    [peak_value_loglin, peak_period_loglin, half_life_loglin] = calculatePeaksAndHalfLives(C_loglin);
    [peak_value_determ, peak_period_determ, half_life_determ] = calculatePeaksAndHalfLives(C_determ);
    
    % Store results in arrays
    peak_values_shock(sector_idx) = peak_value_shock;
    peak_values_loglin(sector_idx) = peak_value_loglin;
    peak_values_determ(sector_idx) = peak_value_determ;
    
    % Calculate amplification (difference between deterministic and loglinear peak values)
    amplifications(sector_idx) = peak_value_determ - peak_value_loglin;
    
    peak_periods_shock(sector_idx) = peak_period_shock;
    peak_periods_loglin(sector_idx) = peak_period_loglin;
    peak_periods_determ(sector_idx) = peak_period_determ;
    
    half_lives_shock(sector_idx) = half_life_shock;
    half_lives_loglin(sector_idx) = half_life_loglin;
    half_lives_determ(sector_idx) = half_life_determ;
    
    % Display progress information
    disp(['  Sector ' num2str(sector_idx) ' - Consumption: peak value=' num2str(peak_value_loglin) ...
          ', peak period=' num2str(peak_period_loglin) ...
          ', half-life=' num2str(half_life_loglin) ...
          ', amplification=' num2str(amplifications(sector_idx))]);
    
    % Save intermediate results periodically
    if mod(sector_idx, 5) == 0
        intermediate_filename = strcat('output/AllSectors_IRS_Intermediate_', save_label, '.mat');
        
        % Store arrays in AllIRS for saving
        AllIRS.peak_values_shock = peak_values_shock;
        AllIRS.peak_values_loglin = peak_values_loglin;
        AllIRS.peak_values_determ = peak_values_determ;
        
        AllIRS.peak_periods_shock = peak_periods_shock;
        AllIRS.peak_periods_loglin = peak_periods_loglin;
        AllIRS.peak_periods_determ = peak_periods_determ;
        
        AllIRS.half_lives_shock = half_lives_shock;
        AllIRS.half_lives_loglin = half_lives_loglin;
        AllIRS.half_lives_determ = half_lives_determ;
        AllIRS.amplifications = amplifications;  % Add amplifications to intermediate saves
        
        save(intermediate_filename, 'AllIRS');
        disp(['  Saved intermediate results after sector ' num2str(sector_idx)]);
    end
end

% Store arrays in AllIRS
AllIRS.peak_values_shock = peak_values_shock;
AllIRS.peak_values_loglin = peak_values_loglin;
AllIRS.peak_values_determ = peak_values_determ;

AllIRS.peak_periods_shock = peak_periods_shock;
AllIRS.peak_periods_loglin = peak_periods_loglin;
AllIRS.peak_periods_determ = peak_periods_determ;

AllIRS.half_lives_shock = half_lives_shock;
AllIRS.half_lives_loglin = half_lives_loglin;
AllIRS.half_lives_determ = half_lives_determ;
AllIRS.amplifications = amplifications;  % Store amplifications array

% Calculate and display average amplification
avg_amplification = mean(amplifications);
disp(['Average amplification across all sectors: ' num2str(avg_amplification)]);

%% SAVE IRS
if save_exper_ind == 1
    filename_Mod = strcat('output/AllSectors_IRS_', save_label, '.mat');
    save(filename_Mod, 'AllIRS');
end

% MODIFIED: We can comment this block since we're now saving all sectors in the AllIRS struct
%% Calculate Half Lives
% Cj_all = zeros(parn_sectors,200);
% for sector_index = 1:parn_sectors 
%     Cj_all(sector_index,:) = (dynare_simul(2 * parn_sectors + sector_index, :) - policies_ss(0*parn_sectors+sector_index));
% end
% shock = IRSLoglinCD{1,1}(1,1:100);
% Cagg_CD = IRSLoglinCD{1,1}(2,1:100);
% Cagg_loglin = IRSLoglin{1,1}(2,1:100);
% Cagg_determ = IRSDeterm{1,1}(2,1:100);
% [peak_shock, half_live_shock] = calculateHalfLives(1-shock);
% disp('Half-lives shock:');
% disp(half_live_shock);
% [peak_CD, half_live_CD] = calculateHalfLives(-Cagg_CD);
% disp('Half-lives CD:');
% disp(half_live_CD);
% [peak_loglin, half_live_loglin] = calculateHalfLives(-Cagg_loglin);
% disp('Half-lives loglin:');
% disp(half_live_loglin);
% [peak_determ, half_live_determ] = calculateHalfLives(-Cagg_determ);
% disp('Half-lives determ:');
% disp(half_live_determ);

% [peak_determ_Cj, half_live_determ_Cj] = calculateHalfLives(-Cj_all);
% disp('Half-lives determ sectoral C:');
% disp(half_live_determ_Cj);
% disp('Peak determ sectoral C:');
% disp(peak_determ_Cj);

% %% 3D GRAPH OF CONSUMPTION REACTION IN ALL SECTORS 
% tic;
% % CONSUMPTION REACTION FROM ALL SECTORS
% Cj_all = zeros(parn_sectors,200);
% time = 0:40;
% for sector_index = 1:parn_sectors 
%     Cj_all(sector_index,:) = (dynare_simul(2 * parn_sectors + sector_index, :) - policies_ss(0*parn_sectors+sector_index));
% end
% Cj_all = Cj_all(:,1:numel(time));
% % TO BE DONE: change 1 by index of s_idx of sector indices
% [x, y] = meshgrid(time, ranking);
% figure;
% surf(x, y, Cj_all);
% xlabel('Time Periods');
% ylabel('Ranking Numbers');
% zlabel('Effect on Consumption');
% title('3D Plot of Consumption vs. Time vs. Ranking');
% set(gca, 'ZDir','reverse')
% 
% % Obtainnig the time period at which consumption peaks (downwards)
% [min_value,min_column] = min(Cj_all,[], 2);
% time_of_peak = min_column-1;
% peak = min_value;
% 
% hold on; 
% for sector_index = 1:parn_sectors
%     plot3(time_of_peak(sector_index), ranking(sector_index), Cj_all(sector_index, time_of_peak(sector_index) + 1), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
% end
% hold off;
% 
% elapsed_time = toc;
% fprintf('It took %.4f seconds to run .\n', elapsed_time);
% disp('FINISHED PLOTTING 3D GRAPH OF CONSUMPTION')


