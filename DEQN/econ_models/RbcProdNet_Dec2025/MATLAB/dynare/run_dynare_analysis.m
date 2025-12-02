function [Results] = run_dynare_analysis(ModData, params, opts)
% RUN_DYNARE_ANALYSIS Runs Dynare analysis including solution, simulation, and IRFs
%
% This function orchestrates all Dynare-based analysis:
%   1. Log-linear (stochastic) solution
%   2. Log-linear simulation with random shocks
%   3. Log-linear impulse responses
%   4. Perfect foresight impulse responses
%   5. Deterministic simulation with random shocks
%
% INPUTS:
%   ModData  - Structure from steady state calibration containing:
%              - parameters: calibrated parameters
%              - policies_ss: steady state policies
%              - endostates_ss: steady state capital (log)
%
%   params   - Structure with model parameters:
%              - n_sectors, IRshock, Sigma_A, etc.
%
%   opts     - Structure with analysis options:
%              - run_loglin_simul: Run log-linear simulation with random shocks (default: true)
%              - run_loglin_irs: Run log-linear IRFs (default: true)
%              - run_determ_irs: Run perfect foresight IRFs (default: true)
%              - run_determ_simul: Run deterministic simulation (default: true)
%              - sector_indices: Sectors to shock for IRFs (default: [1])
%              - modorder: Approximation order for simulation (default: 1)
%              - verbose: Print progress (default: true)
%              - ir_horizon: Horizon for IR calculation (default: 200)
%              - simul_T_loglin: Log-linear simulation length (default: 10000)
%              - simul_T_determ: Deterministic simulation length (default: 1000)
%              - rng_seed: RNG seed for reproducibility (default: [] = current state)
%                          Can be: integer seed, or saved rng_state struct
%
%   NOTE: The stochastic solution (stoch_simul) ALWAYS runs if not already computed.
%         It is required for all log-linear analysis and cannot be skipped.
%
% OUTPUTS:
%   Results  - Structure with all analysis results:
%              - SolData: Solution matrices (A, B, C, D) and steady states
%              - SimulData: Simulation results (log-linear and deterministic)
%              - IRSLoglin: Log-linear impulse responses
%              - IRSDeterm: Perfect foresight impulse responses

%% Input validation
validate_params(params, {'n_sectors', 'IRshock', 'Sigma_A'}, 'run_dynare_analysis');

%% Set default options
if nargin < 3
    opts = struct();
end

opts = set_default(opts, 'run_loglin_simul', true);  % Run log-linear simulation with random shocks
opts = set_default(opts, 'run_loglin_irs', true);
opts = set_default(opts, 'run_determ_irs', true);
opts = set_default(opts, 'run_determ_simul', true);
opts = set_default(opts, 'sector_indices', [1]);
opts = set_default(opts, 'modorder', 1);
opts = set_default(opts, 'verbose', true);
opts = set_default(opts, 'ir_horizon', 200);  % IR calculation horizon
opts = set_default(opts, 'rng_seed', []);  % Empty = use current RNG state; integer = set specific seed
opts = set_default(opts, 'simul_T_loglin', 10000);  % Log-linear simulation length
opts = set_default(opts, 'simul_T_determ', 1000);   % Deterministic simulation length

% Validate sector indices
validate_sector_indices(opts.sector_indices, params.n_sectors, 'run_dynare_analysis');

n_sectors = params.n_sectors;
simul_T_loglin = opts.simul_T_loglin;
simul_T_determ = opts.simul_T_determ;

%% Prepare data for Dynare
if opts.verbose
    fprintf('\n  Preparing Dynare input...\n');
end

policies_ss = ModData.policies_ss;
k_ss = ModData.endostates_ss;
idx = get_variable_indices(n_sectors);
Cagg_ss = exp(policies_ss(idx.cagg - idx.ss_offset));
Lagg_ss = exp(policies_ss(idx.lagg - idx.ss_offset));

% Extract parameters and assign to base workspace for Dynare
params_vars = struct2cell(ModData.parameters);
params_names = fieldnames(ModData.parameters);
for i = 1:numel(params_vars)
    assignin('base', params_names{i}, params_vars{i});
end

% Save ModStruct_temp for Dynare (in dynare folder)
N = opts.ir_horizon;
ax = 0:N-1;

% Get the path to the dynare folder
[dynare_folder, ~, ~] = fileparts(mfilename('fullpath'));
modstruct_path = fullfile(dynare_folder, 'ModStruct_temp.mat');
save(modstruct_path, 'par*', 'policies_ss', 'k_ss', 'N', 'ax', '-regexp', '^par');

% Also save params struct to base workspace for later use
assignin('base', 'params', params);
assignin('base', 'policies_ss', policies_ss);
assignin('base', 'k_ss', k_ss);
assignin('base', 'Cagg_ss', Cagg_ss);
assignin('base', 'Lagg_ss', Lagg_ss);
assignin('base', 'parn_sectors', n_sectors);

%% Initialize Results structure
Results = struct();
Results.params = params;
Results.Cagg_ss = Cagg_ss;
Results.Lagg_ss = Lagg_ss;

%% 1. Stochastic Solution (Log-Linear) - ALWAYS REQUIRED
% Check if we already have valid saved stochastic objects
have_saved_stoch = false;
try
    oo_stoch = evalin('base', 'oo_stoch_');
    if isstruct(oo_stoch.dr) && isfield(oo_stoch.dr, 'ghx')
        have_saved_stoch = true;
    end
catch
    % No saved stochastic objects
end

if ~have_saved_stoch
    if opts.verbose
        fprintf('\n  ── Solving Log-Linear Model (stoch_simul) ──────────────────────\n');
    end
    
    tic;
    
    % Change to dynare folder, run, and return
    current_dir = pwd;
    cd(dynare_folder);
    try
        dynare stoch_simul;
    catch ME
        cd(current_dir);
        rethrow(ME);
    end
    cd(current_dir);
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Log-linear solution completed (%.2f s)\n', elapsed);
    end
    
    % Save stochastic objects (determ_simul will overwrite oo_, M_, options_)
    oo_stoch = evalin('base', 'oo_');
    M_stoch = evalin('base', 'M_');
    options_stoch = evalin('base', 'options_');
    assignin('base', 'oo_stoch_', oo_stoch);
    assignin('base', 'M_stoch_', M_stoch);
    assignin('base', 'options_stoch_', options_stoch);
    
    if opts.verbose
        fprintf('     ✓ Stochastic objects cached\n');
    end
else
    if opts.verbose
        fprintf('\n  ✓ Using cached stochastic solution\n');
    end
end

% Get Dynare objects from saved stochastic solution
oo_ = evalin('base', 'oo_stoch_');
M_ = evalin('base', 'M_stoch_');
options_ = evalin('base', 'options_stoch_');

Results.oo_ = oo_;
Results.M_ = M_;
Results.steady_state = oo_.steady_state;

%% 2. Log-Linear Simulation (with random shocks)
if opts.run_loglin_simul
    if opts.verbose
        fprintf('\n  ── Log-Linear Simulation (T = %d) ────────────────────────────\n', simul_T_loglin);
    end
    
    tic;
    
    % Set RNG seed if provided, otherwise use current state
    if ~isempty(opts.rng_seed)
        if isstruct(opts.rng_seed)
            rng(opts.rng_seed);
        else
            rng(opts.rng_seed);
        end
    end
    
    % Save RNG state for reproducibility
    rng_state = rng;
    Results.rng_state = rng_state;
    if opts.verbose
        fprintf('     RNG: %s (seed = %d)\n', rng_state.Type, rng_state.Seed);
    end
    
    % Generate random shocks
    shockssim = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, simul_T_loglin);
    
    % Simulate using Dynare's simult_
    dynare_simul_loglin = simult_(M_, options_, oo_.steady_state, oo_.dr, shockssim, opts.modorder);
    
    % Extract solution matrices (State Space Representation)
    % S(t) = A*S(t-1) + B*e(t)
    % X(t) = C*S(t-1) + D*e(t)
    SolData = extract_state_space(oo_, M_, n_sectors, policies_ss);
    SolData.shockssim = shockssim;
    
    % Compute simulation statistics
    varlev = exp(dynare_simul_loglin(1:idx.n_dynare,:));
    variables_var = var(dynare_simul_loglin, 0, 2);
    
    shocks_sd = sqrt(var(shockssim, 0, 1)).';
    states_sd = sqrt(variables_var(1:idx.n_states));
    policies_sd = sqrt(variables_var(idx.n_states+1:idx.n_dynare));
    
    SolData.shocks_sd = shocks_sd;
    SolData.states_sd = states_sd;
    SolData.policies_sd = policies_sd;
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed (%.2f s)\n', elapsed);
    end
    
    Results.SolData = SolData;
    Results.SimulLoglin = dynare_simul_loglin;
    
    % Compute model statistics (comparable to HP-filtered empirical targets)
    ModelStats = compute_model_statistics(dynare_simul_loglin, idx, policies_ss, n_sectors);
    Results.ModelStats = ModelStats;
    
    if opts.verbose
        fprintf('\n     Model Statistics (log-linear simulation):\n');
        fprintf('       σ(Y_agg):  %.4f\n', ModelStats.sigma_VA_agg);
        fprintf('       σ(L) avg:  %.4f\n', ModelStats.sigma_L_avg);
        fprintf('       σ(I) avg:  %.4f\n', ModelStats.sigma_I_avg);
    end
end

%% 3. Log-Linear Impulse Responses
if opts.run_loglin_irs
    if opts.verbose
        fprintf('\n  ── Log-Linear IRFs (horizon = %d) ───────────────────────────\n', opts.ir_horizon);
    end
    
    tic;
    IRSLoglin_all = cell(numel(opts.sector_indices), 1);
    
    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        
        % Set shock in initial TFP
        steady_state_shocked = oo_.steady_state;
        steady_state_shocked(n_sectors + sector_idx) = -params.IRshock;
        
        % Zero exogenous shocks (shock is in initial condition)
        shockssim_ir = zeros([opts.ir_horizon, n_sectors]);
        
        % Simulate
        dynare_simul = simult_(M_, options_, steady_state_shocked, oo_.dr, shockssim_ir, opts.modorder);
        
        IRSLoglin_all{ii} = dynare_simul;
        
        if opts.verbose
            fprintf('     • Sector %d\n', sector_idx);
        end
    end
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed %d sectors (%.2f s)\n', numel(opts.sector_indices), elapsed);
    end
    
    Results.IRSLoglin_raw = IRSLoglin_all;
    Results.ir_sector_indices = opts.sector_indices;
end

%% 4. Perfect Foresight Impulse Responses
if opts.run_determ_irs
    if opts.verbose
        fprintf('\n  ── Perfect Foresight IRFs (horizon = %d) ──────────────────────\n', opts.ir_horizon);
    end
    
    tic;
    IRSDeterm_all = cell(numel(opts.sector_indices), 1);
    
    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        
        % Set initial shock
        shocksim_0 = zeros([n_sectors, 1]);
        shocksim_0(sector_idx, 1) = -params.IRshock;
        
        % Prepare shock matrix for perfect foresight (zeros after initial)
        shockssim_ir = zeros([opts.ir_horizon, n_sectors]);
        
        % Save to workspace for Dynare (needed by .mod file)
        assignin('base', 'shocksim_0', shocksim_0);
        assignin('base', 'shockssim_ir', shockssim_ir);
        
        % Run perfect foresight (change to dynare folder)
        current_dir = pwd;
        cd(dynare_folder);
        try
            dynare determ_irs;
        catch ME
            cd(current_dir);
            rethrow(ME);
        end
        cd(current_dir);
        
        % Get results
        Simulated_time_series = evalin('base', 'Simulated_time_series');
        dynare_simul = Simulated_time_series.data';
        
        IRSDeterm_all{ii} = dynare_simul;
        
        if opts.verbose
            fprintf('     • Sector %d\n', sector_idx);
        end
    end
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed %d sectors (%.2f s)\n', numel(opts.sector_indices), elapsed);
    end
    
    Results.IRSDeterm_raw = IRSDeterm_all;
end

%% 5. Deterministic Simulation (Perfect Foresight with Random Shocks)
if opts.run_determ_simul
    if opts.verbose
        fprintf('\n  ── Deterministic Simulation (T = %d) ─────────────────────────\n', simul_T_determ);
    end
    
    tic;
    
    % Generate random shocks for deterministic simulation
    shockssim = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, simul_T_determ);
    
    % Prepare shock matrix for perfect foresight
    % Dynare expects (periods+2 x n_exo): [initial; simulation periods; terminal]
    simul_periods = simul_T_determ - 2;  % Number of periods for perfect_foresight
    shockssim_determ = zeros(simul_T_determ, n_sectors);
    shockssim_determ(2:end-1, :) = shockssim(1:simul_T_determ-2, :);  % Fill simulation periods
    
    if opts.verbose
        fprintf('     Shock matrix: %d × %d | Periods: %d\n', ...
            size(shockssim_determ, 1), size(shockssim_determ, 2), simul_periods);
    end
    
    % Save to workspace (needed by .mod file)
    assignin('base', 'shockssim_determ', shockssim_determ);
    assignin('base', 'simul_periods', simul_periods);
    
    % Update ModStruct_temp with simulation parameters (in dynare folder)
    modstruct_path = fullfile(dynare_folder, 'ModStruct_temp.mat');
    save(modstruct_path, 'shockssim_determ', 'simul_periods', '-append');
    
    % Run deterministic simulation (change to dynare folder)
    current_dir = pwd;
    cd(dynare_folder);
    try
        dynare determ_simul;
        cd(current_dir);
        
        % Get results
        Simulated_time_series = evalin('base', 'Simulated_time_series');
        dynare_simul_determ = Simulated_time_series.data';
        
        Results.SimulDeterm = dynare_simul_determ;
        Results.shockssim_determ = shockssim_determ;
        
        elapsed = toc;
        if opts.verbose
            fprintf('     ✓ Completed (%.2f s)\n', elapsed);
        end
    catch ME
        cd(current_dir);
        if opts.verbose
            fprintf('     ⚠ Failed: %s\n', ME.message);
        end
        Results.SimulDeterm = [];
        Results.determ_simul_error = ME;
    end
end

%% Summary
if opts.verbose
    fprintf('\n  ────────────────────────────────────────────────────────────────\n');
    fprintf('  ✓ Dynare analysis complete\n');
    fprintf('    Results: {%s}\n', strjoin(fieldnames(Results)', ', '));
end

end

%% ==================== Helper Functions ====================

function SolData = extract_state_space(oo_, M_, n_sectors, policies_ss)
% EXTRACT_STATE_SPACE Extract state space representation from Dynare solution
%
% State Space:
%   S(t) = A*S(t-1) + B*e(t)   [state evolution]
%   X(t) = C*S(t-1) + D*e(t)   [policy functions]

    dim = n_sectors;
    
    % Variable indices in Dynare ordering
    k_ind = [1, dim];
    a_ind = [dim+1, 2*dim];
    c_ind = [2*dim+1, 3*dim];
    l_ind = [3*dim+1, 4*dim];
    pk_ind = [4*dim+1, 5*dim];
    pm_ind = [5*dim+1, 6*dim];
    m_ind = [6*dim+1, 7*dim];
    mout_ind = [7*dim+1, 8*dim];
    i_ind = [8*dim+1, 9*dim];
    iout_ind = [9*dim+1, 10*dim];
    p_ind = [10*dim+1, 11*dim];
    q_ind = [11*dim+1, 12*dim];
    y_ind = [12*dim+1, 13*dim];
    cagg_ind = 13*dim+1;
    lagg_ind = 13*dim+2;
    yagg_ind = 13*dim+3;
    iagg_ind = 13*dim+4;
    magg_ind = 13*dim+5;
    
    % Inverse ordering (Dynare internal to model ordering)
    k_ind_inv = [oo_.dr.inv_order_var(k_ind(1)), oo_.dr.inv_order_var(k_ind(2))];
    a_ind_inv = [oo_.dr.inv_order_var(a_ind(1)), oo_.dr.inv_order_var(a_ind(2))];
    c_ind_inv = [oo_.dr.inv_order_var(c_ind(1)), oo_.dr.inv_order_var(c_ind(2))];
    l_ind_inv = [oo_.dr.inv_order_var(l_ind(1)), oo_.dr.inv_order_var(l_ind(2))];
    pk_ind_inv = [oo_.dr.inv_order_var(pk_ind(1)), oo_.dr.inv_order_var(pk_ind(2))];
    pm_ind_inv = [oo_.dr.inv_order_var(pm_ind(1)), oo_.dr.inv_order_var(pm_ind(2))];
    m_ind_inv = [oo_.dr.inv_order_var(m_ind(1)), oo_.dr.inv_order_var(m_ind(2))];
    mout_ind_inv = [oo_.dr.inv_order_var(mout_ind(1)), oo_.dr.inv_order_var(mout_ind(2))];
    i_ind_inv = [oo_.dr.inv_order_var(i_ind(1)), oo_.dr.inv_order_var(i_ind(2))];
    iout_ind_inv = [oo_.dr.inv_order_var(iout_ind(1)), oo_.dr.inv_order_var(iout_ind(2))];
    p_ind_inv = [oo_.dr.inv_order_var(p_ind(1)), oo_.dr.inv_order_var(p_ind(2))];
    q_ind_inv = [oo_.dr.inv_order_var(q_ind(1)), oo_.dr.inv_order_var(q_ind(2))];
    y_ind_inv = [oo_.dr.inv_order_var(y_ind(1)), oo_.dr.inv_order_var(y_ind(2))];
    cagg_ind_inv = oo_.dr.inv_order_var(cagg_ind);
    lagg_ind_inv = oo_.dr.inv_order_var(lagg_ind);
    yagg_ind_inv = oo_.dr.inv_order_var(yagg_ind);
    iagg_ind_inv = oo_.dr.inv_order_var(iagg_ind);
    magg_ind_inv = oo_.dr.inv_order_var(magg_ind);
    
    % State transition matrices
    A = [oo_.dr.ghx(k_ind_inv(1):k_ind_inv(2),:); 
         oo_.dr.ghx(a_ind_inv(1):a_ind_inv(2),:)];
    B = [oo_.dr.ghu(k_ind_inv(1):k_ind_inv(2),:); 
         oo_.dr.ghu(a_ind_inv(1):a_ind_inv(2),:)];
    
    % Policy matrices
    C = [oo_.dr.ghx(c_ind_inv(1):c_ind_inv(2),:);
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
    
    D = [oo_.dr.ghu(c_ind_inv(1):c_ind_inv(2),:);
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
    
    % Store in structure
    SolData = struct();
    SolData.A = A;
    SolData.B = B;
    SolData.C = C;
    SolData.D = D;
    SolData.k_ss = oo_.steady_state(1:dim);
    SolData.policies_ss = policies_ss;
    
    % Store indices for reference
    SolData.indices = struct('k', k_ind, 'a', a_ind, 'c', c_ind, 'l', l_ind, ...
        'pk', pk_ind, 'pm', pm_ind, 'm', m_ind, 'mout', mout_ind, ...
        'i', i_ind, 'iout', iout_ind, 'p', p_ind, 'q', q_ind, 'y', y_ind, ...
        'cagg', cagg_ind, 'lagg', lagg_ind, 'yagg', yagg_ind, ...
        'iagg', iagg_ind, 'magg', magg_ind);
end

function ModelStats = compute_model_statistics(dynare_simul, idx, policies_ss, n_sectors)
% COMPUTE_MODEL_STATISTICS Compute business cycle statistics from Dynare simulation
%
% Dynare simulation output is in log-deviations from steady state.
% These are directly comparable to HP-filtered empirical data 
% (both represent cyclical components around trend/steady state).
%
% INPUTS:
%   dynare_simul - Simulation output (n_vars x T), in log deviations
%   idx          - Variable indices structure from get_variable_indices
%   policies_ss  - Steady state policies (for computing VA weights)
%   n_sectors    - Number of sectors
%
% OUTPUTS:
%   ModelStats - Structure with:
%     - sigma_VA_agg: Aggregate GDP volatility
%     - sigma_L_avg: VA-weighted avg sectoral labor volatility
%     - sigma_I_avg: VA-weighted avg sectoral investment volatility
%     - rho_VA_agg: Aggregate GDP autocorrelation
%     - avg_pairwise_corr_VA: Average pairwise correlation of sectoral VA

    % Extract simulated series (log deviations from SS)
    y_simul = dynare_simul(idx.y(1):idx.y(2), :);      % Sectoral VA
    l_simul = dynare_simul(idx.l(1):idx.l(2), :);      % Sectoral labor
    i_simul = dynare_simul(idx.i(1):idx.i(2), :);      % Sectoral investment
    m_simul = dynare_simul(idx.m(1):idx.m(2), :);      % Sectoral intermediates
    q_simul = dynare_simul(idx.q(1):idx.q(2), :);      % Sectoral gross output
    yagg_simul = dynare_simul(idx.yagg, :);            % Aggregate GDP
    lagg_simul = dynare_simul(idx.lagg, :);            % Aggregate labor
    iagg_simul = dynare_simul(idx.iagg, :);            % Aggregate investment
    magg_simul = dynare_simul(idx.magg, :);            % Aggregate intermediates
    
    % Compute VA-based weights from steady state
    % policies_ss index = Dynare index - ss_offset
    y_ss_idx = (idx.y(1):idx.y(2)) - idx.ss_offset;
    y_ss_log = policies_ss(y_ss_idx);
    y_ss = exp(y_ss_log);
    va_weights = y_ss' / sum(y_ss);
    
    % Compute GO-based weights from steady state (Q = gross output)
    q_ss_idx = (idx.q(1):idx.q(2)) - idx.ss_offset;
    q_ss_log = policies_ss(q_ss_idx);
    q_ss = exp(q_ss_log);
    go_weights = q_ss' / sum(q_ss);
    
    %% ===== AGGREGATE VOLATILITIES (from Dynare aggregate variables) =====
    sigma_VA_agg = std(yagg_simul);
    sigma_L_agg = std(lagg_simul);
    sigma_I_agg = std(iagg_simul);
    sigma_M_agg = std(magg_simul);
    
    % Aggregate GDP autocorrelation
    rho_VA_agg = corr(yagg_simul(1:end-1)', yagg_simul(2:end)');
    
    %% ===== SECTORAL STATISTICS =====
    % Sectoral VA pairwise correlations
    corr_matrix_VA = corr(y_simul');
    upper_tri_idx = triu(true(n_sectors), 1);
    pairwise_corrs = corr_matrix_VA(upper_tri_idx);
    avg_pairwise_corr_VA = mean(pairwise_corrs);
    
    % Sectoral labor volatility (VA-weighted average)
    sigma_L_sectoral = std(l_simul, 0, 2)';
    sigma_L_avg = sum(va_weights .* sigma_L_sectoral);
    
    % Sectoral investment volatility (VA-weighted average)
    sigma_I_sectoral = std(i_simul, 0, 2)';
    sigma_I_avg = sum(va_weights .* sigma_I_sectoral);
    
    %% Store results
    ModelStats = struct();
    
    % Aggregate volatilities
    ModelStats.sigma_VA_agg = sigma_VA_agg;
    ModelStats.sigma_L_agg = sigma_L_agg;
    ModelStats.sigma_I_agg = sigma_I_agg;
    ModelStats.sigma_M_agg = sigma_M_agg;
    
    % Other aggregate moments
    ModelStats.rho_VA_agg = rho_VA_agg;
    ModelStats.avg_pairwise_corr_VA = avg_pairwise_corr_VA;
    
    % Sectoral moments (VA-weighted averages)
    ModelStats.sigma_L_avg = sigma_L_avg;
    ModelStats.sigma_I_avg = sigma_I_avg;
    
    % Full distributions (for diagnostics)
    ModelStats.sigma_L_sectoral = sigma_L_sectoral;
    ModelStats.sigma_I_sectoral = sigma_I_sectoral;
    ModelStats.corr_matrix_VA = corr_matrix_VA;
    ModelStats.va_weights = va_weights;
    ModelStats.go_weights = go_weights;
end

