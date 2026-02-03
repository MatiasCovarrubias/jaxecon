%%% This code runs the dynare codes for Carvalho, Covarrubias and Nuño (2022)
clear; 
clearvars -global;    
clc; 

%% Add paths (must be first to access utility functions)
% Use fullfile with mfilename to ensure we use THIS folder's functions, not cached ones
current_folder = fileparts(mfilename('fullpath'));

% Remove any old RbcProdNet folders from path to avoid loading wrong functions
old_paths = {'RbcProdNet_Oct2025', 'RbcProdNet_nonlinear', 'RbcProdNet_newcalib'};
for i = 1:numel(old_paths)
    paths_to_remove = genpath(fullfile(fileparts(current_folder), old_paths{i}));
    if ~isempty(paths_to_remove)
        rmpath(paths_to_remove);
    end
end

% Clear cached functions to ensure we use the correct versions
clear GraphIRs process_sector_irs process_ir_data run_dynare_analysis;

% Add paths with priority (prepend to path)
addpath(fullfile(current_folder, 'plotting'), '-begin');
addpath(fullfile(current_folder, 'utils'), '-begin');
addpath(fullfile(current_folder, 'dynare'), '-begin');
addpath(fullfile(current_folder, 'steady_state'), '-begin');
addpath(fullfile(current_folder, 'calibration'), '-begin');

% Verify correct GraphIRs is loaded
graphirs_path = which('GraphIRs');
if ~contains(graphirs_path, 'Dec2025')
    warning('GraphIRs loaded from unexpected path: %s', graphirs_path);
end

%% Verify required data files exist
required_files = {'calibration_data.mat', 'TFP_process.mat'};
missing_files = {};
for i = 1:numel(required_files)
    if ~exist(required_files{i}, 'file')
        missing_files{end+1} = required_files{i}; %#ok<SAGROW>
    end
end
if ~isempty(missing_files)
    error('main_IRs:MissingDataFiles', ...
        ['Missing required data files: %s\n' ...
         'Please ensure these files are in the MATLAB path or current directory.'], ...
        strjoin(missing_files, ', '));
end

%% Configuration
config = struct();

% --- Model specification ---
% 'VA'      - Value Added model (TFP multiplies Y, uses modrho/modvcv)
% 'GO'      - Gross Output model (TFP multiplies Q, uses modrho_GO/modvcv_GO)
% 'GO_noVA' - Gross Output without VA data (uses modrho_GO_noVA/modvcv_GO_noVA)
config.model_type = 'VA';

% --- General settings ---
config.save_results = false;       % Save data and graphs
config.recalibrate = true;         % Recalibrate steady state (false = load saved)
config.compute_all_sectors = false; % true = all 37 sectors, false = specified sectors
config.big_sector_ind = false;     % Only used when compute_all_sectors = false

% --- Experiment labels ---
config.date = "_December_2025";
config.exp_label = "_nonlinear_Min";

% --- Dynare analysis settings ---
% First-order (linear) approximation: uses stoch_simul.mod with order=1
% Second-order (quadratic) approximation: uses stoch_simul_2ndOrder.mod with order=2
% Perfect foresight (nonlinear): uses Newton solver
config.run_firstorder_simul = true;   % Run first-order (linear) simulation
config.run_secondorder_simul = true;  % Run second-order simulation (captures precautionary effects)
config.run_pf_simul = true;           % Run perfect foresight simulation (slower)
config.run_firstorder_irs = true;     % Compute first-order IRFs
config.run_secondorder_irs = true;    % Compute second-order IRFs (asymmetric responses)
config.run_pf_irs = true;             % Compute perfect foresight IRFs (slowest)

% --- IR settings ---
config.ir_horizon = 200;           % Horizon for IR calculation (needs to be long for convergence)
config.ir_plot_length = 60;        % Periods to plot in IR figures
config.plot_irs = true;            % Plot IRF figures (set false for batch runs)

% --- Simulation settings ---
config.simul_T_firstorder = 2500;   % First-order (linear) simulation length (fast)
config.simul_T_secondorder = 2500;  % Second-order (quadratic) simulation length (fast)
config.simul_T_pf = 500;            % Perfect foresight simulation length (slower)

% --- Shock values configuration ---
% Shock convention: IRshock is defined such that the TFP deviation is -IRshock.
%   - For A to drop to A_neg:  IRshock = -log(A_neg)
%   - For symmetric positive:  IRshock = log(A_neg) = -log(1/A_neg)
%
% SYMMETRIC SHOCKS: For a negative shock A = A_neg, the symmetric positive shock
% gives A = 1/A_neg (same magnitude in log space, opposite sign).
%   Example: A_neg = 0.8 → A_pos = 1/0.8 = 1.25 (NOT 1.2!)
%
% TFP process: a_{t+1} = rho * a_t + e_t, where a = log(A) - log(A_ss)
% At steady state: a = 0 (A = A_ss = 1 normalized)
% Initial condition: a_0 = shock (when starting from SS)

A_neg_20pct = 0.80;   % A drops to 0.80 (-20%)
A_neg_5pct  = 0.95;   % A drops to 0.95 (-5%)

config.shock_values = [
    struct('value', -log(A_neg_20pct),    'label', 'neg20pct', 'description', sprintf('-%.0f%% TFP (A=%.2f)', (1-A_neg_20pct)*100, A_neg_20pct));
    struct('value',  log(A_neg_20pct),    'label', 'pos20pct', 'description', sprintf('+%.0f%% TFP (A=%.2f, symmetric)', (1/A_neg_20pct-1)*100, 1/A_neg_20pct));
    struct('value', -log(A_neg_5pct),     'label', 'neg5pct',  'description', sprintf('-%.0f%% TFP (A=%.2f)', (1-A_neg_5pct)*100, A_neg_5pct));
    struct('value',  log(A_neg_5pct),     'label', 'pos5pct',  'description', sprintf('+%.1f%% TFP (A=%.4f, symmetric)', (1/A_neg_5pct-1)*100, 1/A_neg_5pct))
];

% --- Target elasticities of substitution ---
config.sigma_c = 0.5;
config.sigma_m = 0.01;
config.sigma_q = 0.5;
config.sigma_y = 0.8;
config.sigma_I = 0.5;
config.sigma_l = 0.1;

%% Derived settings
N_SECTORS = 37;  % Model constant

if config.compute_all_sectors
    sector_indices = 1:N_SECTORS;
elseif config.big_sector_ind
    sector_indices = [20, 24];
else
    sector_indices = [1];
end

% Validate sector indices
validate_sector_indices(sector_indices, N_SECTORS, 'main_IRs');

save_label = strcat(config.date, config.exp_label);

% Verify shock values (print what A levels will be achieved)
fprintf('\n  Shock configuration verification:\n');
for i = 1:numel(config.shock_values)
    shock = config.shock_values(i);
    A_level = exp(-shock.value);  % TFP deviation = -IRshock, so A = exp(-IRshock)
    fprintf('    %s: IRshock=%.4f → A=%.4f (%s)\n', ...
        shock.label, shock.value, A_level, shock.description);
end

%% Set up experiment folder structure
exp_paths = setup_experiment_folder(save_label);
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                      EXPERIMENT CONFIGURATION                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');
fprintf('  Experiment:  %s\n', save_label);
fprintf('  Folder:      %s\n', exp_paths.experiment);
fprintf('  Sectors:     [%s]\n', num2str(sector_indices));

%% Initialize params
params = struct();
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.33;
params.theta = 1;
params.phi = 4;

%% Load calibration data
[calib_data, params] = load_calibration_data(params, sector_indices, config.model_type);
labels = calib_data.labels;
fprintf('\n  ✓ Calibration data loaded (model_type: %s)\n', config.model_type);

% Display empirical targets
emp_tgt = calib_data.empirical_targets;
fprintf('\n  Empirical Targets (HP-filtered, λ=%d, %s aggregation):\n', ...
    emp_tgt.hp_lambda, emp_tgt.aggregation_method);
fprintf('    ── Aggregate volatilities ──\n');
fprintf('    σ(Y_agg):         %.4f   (aggregate GDP, Törnqvist)\n', emp_tgt.sigma_VA_agg);
fprintf('    σ(L_agg):         %.4f   (aggregate labor, simple sum)\n', emp_tgt.sigma_L_agg);
fprintf('    σ(I_agg):         %.4f   (aggregate investment, Törnqvist)\n', emp_tgt.sigma_I_agg);
fprintf('    ── Average sectoral volatilities ──\n');
fprintf('    σ(L) avg:         %.4f   (VA-weighted avg of sectoral labor vol)\n', emp_tgt.sigma_L_avg);
fprintf('    σ(I) avg:         %.4f   (VA-weighted avg of sectoral investment vol)\n', emp_tgt.sigma_I_avg);

%% Steady State Calibration
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                     STEADY STATE CALIBRATION                        ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');

if config.recalibrate
    params.sigma_c = config.sigma_c;
    params.sigma_m = config.sigma_m;
    params.sigma_q = config.sigma_q;
    params.sigma_y = config.sigma_y;
    params.sigma_I = config.sigma_I;
    params.sigma_l = config.sigma_l;
    
    fprintf('  Target elasticities:\n');
    fprintf('    σ_c = %.2f    σ_m = %.2f    σ_q = %.2f\n', params.sigma_c, params.sigma_m, params.sigma_q);
    fprintf('    σ_y = %.2f    σ_I = %.2f    σ_l = %.2f\n', params.sigma_y, params.sigma_I, params.sigma_l);
    fprintf('\n');
    
    calib_opts = struct();
    calib_opts.gridpoints = 8;
    calib_opts.verbose = true;
    calib_opts.sol_guess_file = 'SS_CDsolution_norm_permanent.mat';
    calib_opts.fsolve_options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,...
        'MaxFunEvals',10000000,'MaxIter',10000);
    
    tic;
    [ModData, params] = calibrate_steady_state(params, calib_opts);
    elapsed_calib = toc;
    
    ss_file = 'calibrated_steady_state.mat';
    save(ss_file, 'ModData', 'params');
    fprintf('\n  ✓ Steady state saved: %s\n', ss_file);
    fprintf('  ✓ Total calibration time: %.2f seconds\n', elapsed_calib);
else
    ss_file = 'calibrated_steady_state.mat';
    if exist(ss_file, 'file')
        load(ss_file, 'ModData', 'params');
        fprintf('  ✓ Loaded saved steady state: %s\n', ss_file);
    else
        error('main:MissingSteadyState', ...
            ['Steady state file not found: %s\n' ...
             'Set config.recalibrate = true to compute it.'], ss_file);
    end
end

%% Run Dynare Analysis
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                        DYNARE ANALYSIS                              ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');

n_shocks = numel(config.shock_values);
run_any_irs = config.run_firstorder_irs || config.run_secondorder_irs || config.run_pf_irs;

% Storage for results
AllShockResults = struct();
AllShockResults.shock_configs = config.shock_values;
AllShockResults.n_shocks = n_shocks;
AllShockResults.DynareResults = cell(n_shocks, 1);
AllShockResults.IRFResults = cell(n_shocks, 1);
BaseResults = struct();

% Run base simulation (stoch_simul and optionally pf_simul)
if config.run_firstorder_simul || config.run_secondorder_simul || config.run_pf_simul
    dynare_opts_base = struct();
    dynare_opts_base.run_firstorder_simul = config.run_firstorder_simul;
    dynare_opts_base.run_secondorder_simul = config.run_secondorder_simul;
    dynare_opts_base.run_firstorder_irs = false;
    dynare_opts_base.run_pf_irs = false;
    dynare_opts_base.run_pf_simul = config.run_pf_simul;
    dynare_opts_base.sector_indices = sector_indices;
    dynare_opts_base.verbose = true;
    dynare_opts_base.ir_horizon = config.ir_horizon;
    dynare_opts_base.simul_T_firstorder = config.simul_T_firstorder;
    dynare_opts_base.simul_T_secondorder = config.simul_T_secondorder;
    dynare_opts_base.simul_T_pf = config.simul_T_pf;
    dynare_opts_base.model_type = config.model_type;
    
    params.IRshock = config.shock_values(1).value;
    
    fprintf('\n  ┌─ Simulation Configuration ────────────────────────────────────┐\n');
    fprintf('  │  Approximation methods:                                        │\n');
    if config.run_firstorder_simul
        fprintf('  │    • First-order (linear):    ON  (T = %d)                 │\n', config.simul_T_firstorder);
    else
        fprintf('  │    • First-order (linear):    OFF                            │\n');
    end
    if config.run_secondorder_simul
        fprintf('  │    • Second-order (quadratic):ON  (T = %d)                 │\n', config.simul_T_secondorder);
    else
        fprintf('  │    • Second-order (quadratic):OFF                            │\n');
    end
    if config.run_pf_simul
        fprintf('  │    • Perfect foresight (NL):  ON  (T = %d)                  │\n', config.simul_T_pf);
    else
        fprintf('  │    • Perfect foresight (NL):  OFF                            │\n');
    end
    fprintf('  └────────────────────────────────────────────────────────────────┘\n\n');
    
    tic;
    BaseResults = run_dynare_analysis(ModData, params, dynare_opts_base);
    elapsed_base = toc;
    
    % Check what actually ran
    has_1storder = isfield(BaseResults, 'SimulFirstOrder') && ~isempty(BaseResults.SimulFirstOrder);
    has_2ndorder = isfield(BaseResults, 'SimulSecondOrder') && ~isempty(BaseResults.SimulSecondOrder);
    has_pf = isfield(BaseResults, 'SimulPerfectForesight') && ~isempty(BaseResults.SimulPerfectForesight);
    
    fprintf('\n  Simulation results:\n');
    if has_1storder, fprintf('    ✓ First-order (linear) computed\n'); else, fprintf('    ✗ First-order skipped\n'); end
    if has_2ndorder, fprintf('    ✓ Second-order (quadratic) computed\n'); else, fprintf('    ✗ Second-order skipped\n'); end
    if has_pf, fprintf('    ✓ Perfect foresight (nonlinear) computed\n'); else, fprintf('    ✗ Perfect foresight skipped\n'); end
    if isfield(BaseResults, 'pf_simul_error')
        fprintf('    ⚠ Perfect foresight simulation failed: %s\n', BaseResults.pf_simul_error.message);
    end
    fprintf('    • Elapsed time: %.2f seconds\n', elapsed_base);
end

% Run IRF analysis for each shock value (if enabled)
if run_any_irs
    fprintf('\n  ┌─ Impulse Response Analysis ──────────────────────────────────┐\n');
    fprintf('  │  Computing IRFs for %d shock configurations                   │\n', n_shocks);
    fprintf('  │  Sectors: [%s]                                                │\n', num2str(sector_indices));
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
    
    for shock_idx = 1:n_shocks
        shock_config = config.shock_values(shock_idx);
        params.IRshock = shock_config.value;
        
        fprintf('\n  ── Shock %d/%d: %s ──────────────────────────────\n', ...
            shock_idx, n_shocks, shock_config.description);
        fprintf('     Shock value: %.4f\n', shock_config.value);
        
        dynare_opts = struct();
        dynare_opts.run_firstorder_simul = false;
        dynare_opts.run_firstorder_irs = config.run_firstorder_irs;
        dynare_opts.run_secondorder_irs = config.run_secondorder_irs;
        dynare_opts.run_pf_irs = config.run_pf_irs;
        dynare_opts.run_pf_simul = false;
        dynare_opts.sector_indices = sector_indices;
        dynare_opts.verbose = true;
        dynare_opts.ir_horizon = config.ir_horizon;
        dynare_opts.simul_T_firstorder = config.simul_T_firstorder;
        dynare_opts.simul_T_pf = config.simul_T_pf;
        dynare_opts.model_type = config.model_type;
        
        tic;
        DynareResults = run_dynare_analysis(ModData, params, dynare_opts);
        elapsed_irf = toc;
        
        % Copy simulation data from base run (if available)
        if isfield(BaseResults, 'SolData')
            DynareResults.SolData = BaseResults.SolData;
        end
        if isfield(BaseResults, 'SimulFirstOrder')
            DynareResults.SimulFirstOrder = BaseResults.SimulFirstOrder;
        end
        if isfield(BaseResults, 'SimulPerfectForesight')
            DynareResults.SimulPerfectForesight = BaseResults.SimulPerfectForesight;
        end
        if isfield(BaseResults, 'shockssim_pf')
            DynareResults.shockssim_pf = BaseResults.shockssim_pf;
        end
        if isfield(BaseResults, 'rng_state')
            DynareResults.rng_state = BaseResults.rng_state;
        end
        
        AllShockResults.DynareResults{shock_idx} = DynareResults;
        
        % Copy shared data to BaseResults if not already present (from first IRF run)
        if shock_idx == 1
            if ~isfield(BaseResults, 'TheoStats') && isfield(DynareResults, 'TheoStats')
                BaseResults.TheoStats = DynareResults.TheoStats;
            end
            if ~isfield(BaseResults, 'Cagg_ss') && isfield(DynareResults, 'Cagg_ss')
                BaseResults.Cagg_ss = DynareResults.Cagg_ss;
                BaseResults.Lagg_ss = DynareResults.Lagg_ss;
            end
        end
        
        % Process IRFs for this shock
        ir_opts = struct();
        ir_opts.plot_graphs = config.plot_irs && ~config.compute_all_sectors;  % Plot for each shock (not just first)
        ir_opts.save_graphs = config.save_results;
        ir_opts.save_intermediate = config.save_results && config.compute_all_sectors;
        ir_opts.save_interval = 5;
        ir_opts.exp_paths = exp_paths;
        ir_opts.save_label = shock_config.label;
        ir_opts.ir_plot_length = config.ir_plot_length;
        ir_opts.shock_description = shock_config.description;
        
        IRFResults = process_sector_irs(DynareResults, params, ModData, labels, ir_opts);
        IRFResults.shock_config = shock_config;
        AllShockResults.IRFResults{shock_idx} = IRFResults;
        
        fprintf('     ✓ Completed in %.2f seconds\n', elapsed_irf);
    end
    
    fprintf('\n  ✓ All %d shock configurations processed\n', n_shocks);
else
    fprintf('\n  ⊘ IRF analysis skipped (disabled in config)\n');
end

%% Build ModelData structure (core model - lightweight)
ModelData = struct();

% Metadata
ModelData.metadata.date = config.date;
ModelData.metadata.exp_label = config.exp_label;
ModelData.metadata.save_label = save_label;
ModelData.metadata.model_type = config.model_type;
ModelData.metadata.sector_indices = sector_indices;
ModelData.metadata.sector_labels = labels.sector_labels;
ModelData.metadata.config = config;
ModelData.metadata.exp_paths = exp_paths;

% Shock configuration metadata
ModelData.metadata.shock_configs = config.shock_values;
ModelData.metadata.n_shocks = n_shocks;
shock_labels_meta = cell(n_shocks, 1);
shock_values_meta = zeros(n_shocks, 1);
for i = 1:n_shocks
    shock_labels_meta{i} = config.shock_values(i).label;
    shock_values_meta(i) = config.shock_values(i).value;
end
ModelData.metadata.shock_labels = shock_labels_meta;
ModelData.metadata.shock_values = shock_values_meta;

% Calibration
ModelData.calibration = calib_data;
ModelData.params = params;

% Empirical targets (always available, independent of simulation)
ModelData.EmpiricalTargets = calib_data.empirical_targets;

% Steady State (always available from ModData)
ModelData.SteadyState.parameters = ModData.parameters;
ModelData.SteadyState.policies_ss = ModData.policies_ss;
ModelData.SteadyState.endostates_ss = ModData.endostates_ss;
ModelData.SteadyState.Cagg_ss = ModData.Cagg_ss;
ModelData.SteadyState.Lagg_ss = ModData.Lagg_ss;
ModelData.SteadyState.Yagg_ss = ModData.Yagg_ss;
ModelData.SteadyState.Iagg_ss = ModData.Iagg_ss;
ModelData.SteadyState.Magg_ss = ModData.Magg_ss;
ModelData.SteadyState.V_ss = ModData.V_ss;

% Solution (from base results, if simulation was run)
if isfield(BaseResults, 'SolData')
    ModelData.Solution.StateSpace.A = BaseResults.SolData.A;
    ModelData.Solution.StateSpace.B = BaseResults.SolData.B;
    ModelData.Solution.StateSpace.C = BaseResults.SolData.C;
    ModelData.Solution.StateSpace.D = BaseResults.SolData.D;
    ModelData.Solution.indices = BaseResults.SolData.indices;
end
if isfield(BaseResults, 'steady_state')
    ModelData.Solution.steady_state = BaseResults.steady_state;
end

% Theoretical statistics (from state-space solution, always available after stoch_simul)
if isfield(BaseResults, 'TheoStats') && ~isempty(fieldnames(BaseResults.TheoStats))
    ModelData.Statistics.TheoStats = BaseResults.TheoStats;
end

% Simulation statistics only (NOT full simulations - those go in ModelData_simulation)
if isfield(BaseResults, 'SolData') && isfield(BaseResults.SolData, 'shocks_sd')
    ModelData.Statistics.shocks_sd = BaseResults.SolData.shocks_sd;
    ModelData.Statistics.states_sd = BaseResults.SolData.states_sd;
    ModelData.Statistics.policies_sd = BaseResults.SolData.policies_sd;
end

%% Build ModelData_simulation structure (full simulation data - heavy)
% This structure holds full simulation time series only.
% Summary statistics go in ModelData.Statistics (lightweight).
ModelData_simulation = struct();
ModelData_simulation.metadata.save_label = save_label;
ModelData_simulation.metadata.exp_paths = exp_paths;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                       SIMULATION RESULTS                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');

has_1storder_simul = false;
has_2ndorder_simul = false;
has_pf_simul = false;

n = params.n_sectors;
idx = get_variable_indices(n);

%% ===== FIRST-ORDER (Linear) Simulation =====
if isfield(BaseResults, 'SimulFirstOrder') && ~isempty(BaseResults.SimulFirstOrder)
    has_1storder_simul = true;
    simul = BaseResults.SimulFirstOrder;  % n_vars × T (log deviations from SS)
    
    % === ModelData_simulation: Full time series only ===
    ModelData_simulation.FirstOrder.full_simul = simul;
    ModelData_simulation.FirstOrder.shocks = BaseResults.SolData.shockssim;
    ModelData_simulation.FirstOrder.variable_indices = idx;
    
    % === ModelData.Statistics: Summary statistics ===
    % States: [k; a] (indices 1:2n)
    states_simul = simul(1:idx.n_states, :);
    % Policies: remaining variables (indices 2n+1:end)
    policies_simul = simul(idx.n_states+1:end, :);
    
    ModelData.Statistics.FirstOrder.states_mean = mean(states_simul, 2);
    ModelData.Statistics.FirstOrder.states_std = std(states_simul, 0, 2);
    ModelData.Statistics.FirstOrder.policies_mean = mean(policies_simul, 2);
    ModelData.Statistics.FirstOrder.policies_std = std(policies_simul, 0, 2);
    
    % Business cycle statistics (detailed)
    if isfield(BaseResults, 'ModelStats')
        ModelData.Statistics.FirstOrder.ModelStats = BaseResults.ModelStats;
    end
    
    fprintf('\n  First-Order (Linear) Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', size(simul, 1), size(simul, 2));
    fprintf('    States:   mean=%.2e, std=%.4f (avg)\n', ...
        mean(abs(ModelData.Statistics.FirstOrder.states_mean)), ...
        mean(ModelData.Statistics.FirstOrder.states_std));
    fprintf('    Policies: mean=%.2e, std=%.4f (avg)\n', ...
        mean(abs(ModelData.Statistics.FirstOrder.policies_mean)), ...
        mean(ModelData.Statistics.FirstOrder.policies_std));
end

%% ===== SECOND-ORDER (Quadratic) Simulation =====
if isfield(BaseResults, 'SimulSecondOrder') && ~isempty(BaseResults.SimulSecondOrder)
    has_2ndorder_simul = true;
    simul = BaseResults.SimulSecondOrder;  % n_vars × T (log deviations from SS)
    
    % === ModelData_simulation: Full time series only ===
    ModelData_simulation.SecondOrder.full_simul = simul;
    if isfield(BaseResults, 'shockssim_2nd')
        ModelData_simulation.SecondOrder.shocks = BaseResults.shockssim_2nd;
    end
    ModelData_simulation.SecondOrder.variable_indices = idx;
    
    % === ModelData.Statistics: Summary statistics ===
    states_simul = simul(1:idx.n_states, :);
    policies_simul = simul(idx.n_states+1:end, :);
    
    ModelData.Statistics.SecondOrder.states_mean = mean(states_simul, 2);
    ModelData.Statistics.SecondOrder.states_std = std(states_simul, 0, 2);
    ModelData.Statistics.SecondOrder.policies_mean = mean(policies_simul, 2);
    ModelData.Statistics.SecondOrder.policies_std = std(policies_simul, 0, 2);
    
    % Business cycle statistics (detailed)
    if isfield(BaseResults, 'ModelStats2nd')
        ModelData.Statistics.SecondOrder.ModelStats = BaseResults.ModelStats2nd;
    end
    
    fprintf('\n  Second-Order (Quadratic) Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', size(simul, 1), size(simul, 2));
    fprintf('    States:   mean=%.4f, std=%.4f (avg)\n', ...
        mean(ModelData.Statistics.SecondOrder.states_mean), ...
        mean(ModelData.Statistics.SecondOrder.states_std));
    fprintf('    Policies: mean=%.4f, std=%.4f (avg)\n', ...
        mean(ModelData.Statistics.SecondOrder.policies_mean), ...
        mean(ModelData.Statistics.SecondOrder.policies_std));
    
    % Compare means to first-order (precautionary effects)
    if has_1storder_simul
        mean_diff_states = ModelData.Statistics.SecondOrder.states_mean - ...
                           ModelData.Statistics.FirstOrder.states_mean;
        mean_diff_policies = ModelData.Statistics.SecondOrder.policies_mean - ...
                             ModelData.Statistics.FirstOrder.policies_mean;
        fprintf('    Precautionary shift: states=%.4f, policies=%.4f (avg mean diff)\n', ...
            mean(mean_diff_states), mean(mean_diff_policies));
    end
end

% Display Model vs Empirical comparison using THEORETICAL moments
if isfield(BaseResults, 'TheoStats') && ~isempty(fieldnames(BaseResults.TheoStats))
    theo_stats = BaseResults.TheoStats;
    emp_tgt = calib_data.empirical_targets;
    
    fprintf('\n  ┌─ Model vs Empirical Comparison ────────────────────────────────┐\n');
    fprintf('  │                                    Model      Empirical  Ratio │\n');
    fprintf('  │  ── Aggregate volatilities (theoretical) ─────────────────────  │\n');
    fprintf('  │  σ(Y_agg):                        %6.4f      %6.4f    %5.2f │\n', ...
        theo_stats.sigma_VA_agg, emp_tgt.sigma_VA_agg, ...
        theo_stats.sigma_VA_agg / emp_tgt.sigma_VA_agg);
    fprintf('  │  σ(I_agg):                        %6.4f      %6.4f    %5.2f │\n', ...
        theo_stats.sigma_I_agg, emp_tgt.sigma_I_agg, ...
        theo_stats.sigma_I_agg / emp_tgt.sigma_I_agg);
    
    % Labor uses simulation-based headcount (model's lagg is CES, not comparable to data)
    if isfield(BaseResults, 'ModelStats')
        model_stats = BaseResults.ModelStats;
        fprintf('  │  ── Labor aggregate (headcount, from simulation) ─────────────  │\n');
        fprintf('  │  σ(L_hc):                         %6.4f      %6.4f    %5.2f │\n', ...
            model_stats.sigma_L_hc_agg, emp_tgt.sigma_L_agg, ...
            model_stats.sigma_L_hc_agg / emp_tgt.sigma_L_agg);
        fprintf('  │  ── Average sectoral volatilities (from simulation) ────────── │\n');
        fprintf('  │  σ(L) avg:                        %6.4f      %6.4f    %5.2f │\n', ...
            model_stats.sigma_L_avg, emp_tgt.sigma_L_avg, ...
            model_stats.sigma_L_avg / emp_tgt.sigma_L_avg);
        fprintf('  │  σ(I) avg:                        %6.4f      %6.4f    %5.2f │\n', ...
            model_stats.sigma_I_avg, emp_tgt.sigma_I_avg, ...
            model_stats.sigma_I_avg / emp_tgt.sigma_I_avg);
    end
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
end

% RNG state for reproducibility
if isfield(BaseResults, 'rng_state')
    ModelData_simulation.rng_state = BaseResults.rng_state;
end

%% ===== PERFECT FORESIGHT (Nonlinear) Simulation =====
if isfield(BaseResults, 'SimulPerfectForesight') && ~isempty(BaseResults.SimulPerfectForesight)
    has_pf_simul = true;
    simul = BaseResults.SimulPerfectForesight;  % n_vars × T (log deviations from SS)
    
    % === ModelData_simulation: Full time series only ===
    ModelData_simulation.PerfectForesight.full_simul = simul;
    if isfield(BaseResults, 'shockssim_determ')
        ModelData_simulation.PerfectForesight.shocks = BaseResults.shockssim_determ;
    end
    ModelData_simulation.PerfectForesight.variable_indices = idx;
    
    % === ModelData.Statistics: Summary statistics ===
    states_simul = simul(1:idx.n_states, :);
    policies_simul = simul(idx.n_states+1:end, :);
    
    ModelData.Statistics.PerfectForesight.states_mean = mean(states_simul, 2);
    ModelData.Statistics.PerfectForesight.states_std = std(states_simul, 0, 2);
    ModelData.Statistics.PerfectForesight.policies_mean = mean(policies_simul, 2);
    ModelData.Statistics.PerfectForesight.policies_std = std(policies_simul, 0, 2);
    
    fprintf('\n  Perfect Foresight (Nonlinear) Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', size(simul, 1), size(simul, 2));
    fprintf('    States:   mean=%.4f, std=%.4f (avg)\n', ...
        mean(ModelData.Statistics.PerfectForesight.states_mean), ...
        mean(ModelData.Statistics.PerfectForesight.states_std));
    fprintf('    Policies: mean=%.4f, std=%.4f (avg)\n', ...
        mean(ModelData.Statistics.PerfectForesight.policies_mean), ...
        mean(ModelData.Statistics.PerfectForesight.policies_std));
    
    % Compare all three methods
    if has_1storder_simul || has_2ndorder_simul
        fprintf('\n    ── Comparison across methods (avg over all variables) ──\n');
        fprintf('                         Mean(states)   Std(states)   Mean(pol)   Std(pol)\n');
        if has_1storder_simul
            fprintf('    First-Order:         %11.4f   %11.4f   %9.4f   %9.4f\n', ...
                mean(ModelData.Statistics.FirstOrder.states_mean), ...
                mean(ModelData.Statistics.FirstOrder.states_std), ...
                mean(ModelData.Statistics.FirstOrder.policies_mean), ...
                mean(ModelData.Statistics.FirstOrder.policies_std));
        end
        if has_2ndorder_simul
            fprintf('    Second-Order:        %11.4f   %11.4f   %9.4f   %9.4f\n', ...
                mean(ModelData.Statistics.SecondOrder.states_mean), ...
                mean(ModelData.Statistics.SecondOrder.states_std), ...
                mean(ModelData.Statistics.SecondOrder.policies_mean), ...
                mean(ModelData.Statistics.SecondOrder.policies_std));
        end
        fprintf('    Perfect Foresight:   %11.4f   %11.4f   %9.4f   %9.4f\n', ...
            mean(ModelData.Statistics.PerfectForesight.states_mean), ...
            mean(ModelData.Statistics.PerfectForesight.states_std), ...
            mean(ModelData.Statistics.PerfectForesight.policies_mean), ...
            mean(ModelData.Statistics.PerfectForesight.policies_std));
    end
    
    %% Capital Preallocation Analysis (Perfect Foresight)
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                   CAPITAL PREALLOCATION ANALYSIS                    ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════════╝\n');
    
    prealloc_opts = struct();
    prealloc_opts.plot_figure = true;
    prealloc_opts.save_figure = config.save_results;
    prealloc_opts.figures_folder = exp_paths.figures;
    prealloc_opts.save_label = save_label;
    prealloc_opts.highlighted_sector = sector_indices(1);  % Mining/Oil/Gas (sector 1)
    
    CapitalStats = analyze_capital_preallocation(BaseResults.SimulPerfectForesight, params, prealloc_opts, ModData);
    ModelData_simulation.PerfectForesight.CapitalStats = CapitalStats;
end

%% Build ModelData_IRs structure (IRF data)
ModelData_IRs = struct();
ModelData_IRs.metadata.save_label = save_label;
ModelData_IRs.metadata.exp_paths = exp_paths;

has_irfs = false;

if run_any_irs && ~isempty(AllShockResults.IRFResults{1})
    has_irfs = true;
    
    ModelData_IRs.shock_configs = config.shock_values;
    ModelData_IRs.n_shocks = n_shocks;
    ModelData_IRs.sector_indices = sector_indices;
    ModelData_IRs.ir_horizon = config.ir_horizon;
    
    ModelData_IRs.by_shock = AllShockResults.IRFResults;
    ModelData_IRs.by_label = struct();
    for i = 1:n_shocks
        label = matlab.lang.makeValidName(config.shock_values(i).label);
        ModelData_IRs.by_label.(label) = AllShockResults.IRFResults{i};
    end
    
    % Print IRF Summary
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                         IRF SUMMARY                                 ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
    
    % Check if second-order IRFs are available
    has_2ndorder_irfs = isfield(AllShockResults.IRFResults{1}.Statistics, 'peak_values_secondorder') && ...
        any(AllShockResults.IRFResults{1}.Statistics.peak_values_secondorder > 0);
    
    if has_2ndorder_irfs
        fprintf('  %-12s %6s %10s %10s %10s %10s\n', 'Shock', 'A₀', 'Peak(1st)', 'Peak(2nd)', 'Peak(PF)', 'Amplif(PF)');
        fprintf('  %s\n', repmat('─', 1, 66));
        for i = 1:n_shocks
            irf_res = AllShockResults.IRFResults{i};
            shock_cfg = config.shock_values(i);
            A_level = exp(-shock_cfg.value);  % Initial TFP level
            avg_peak_1st = mean(irf_res.Statistics.peak_values_firstorder);
            avg_peak_2nd = mean(irf_res.Statistics.peak_values_secondorder);
            avg_peak_pf = mean(irf_res.Statistics.peak_values_pf);
            avg_amplif_rel = mean(irf_res.Statistics.amplifications_rel);
            fprintf('  %-12s %6.2f %10.4f %10.4f %10.4f %9.1f%%\n', ...
                shock_cfg.label, A_level, avg_peak_1st, avg_peak_2nd, avg_peak_pf, avg_amplif_rel);
        end
        fprintf('  %s\n', repmat('─', 1, 66));
        fprintf('  1st = First-Order, 2nd = Second-Order, PF = Perfect Foresight\n');
        fprintf('  Peak = |max consumption deviation|, Amplif = (PF/1st - 1) × 100\n');
    else
        fprintf('  %-12s %6s %10s %10s %10s\n', 'Shock', 'A₀', 'Peak(1st)', 'Peak(PF)', 'Amplif');
        fprintf('  %s\n', repmat('─', 1, 52));
        for i = 1:n_shocks
            irf_res = AllShockResults.IRFResults{i};
            shock_cfg = config.shock_values(i);
            A_level = exp(-shock_cfg.value);  % Initial TFP level
            avg_peak_1st = mean(irf_res.Statistics.peak_values_firstorder);
            avg_peak_pf = mean(irf_res.Statistics.peak_values_pf);
            avg_amplif_rel = mean(irf_res.Statistics.amplifications_rel);
            fprintf('  %-12s %6.2f %10.4f %10.4f %9.1f%%\n', ...
                shock_cfg.label, A_level, avg_peak_1st, avg_peak_pf, avg_amplif_rel);
        end
        fprintf('  %s\n', repmat('─', 1, 52));
        fprintf('  1st = First-Order, PF = Perfect Foresight\n');
        fprintf('  Peak = |max consumption deviation|, Amplif = (PF/1st - 1) × 100\n');
    end
end

%% Save results
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                        SAVING RESULTS                               ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');

if config.save_results
    % Save ModelData (core - lightweight)
    filename_model = fullfile(exp_paths.experiment, 'ModelData.mat');
    save(filename_model, 'ModelData');
    fprintf('\n  ✓ ModelData saved: %s\n', filename_model);
    
    % Save ModelData_simulation (full simulations - heavy)
    if has_1storder_simul || has_2ndorder_simul || has_pf_simul
        filename_simul = fullfile(exp_paths.experiment, 'ModelData_simulation.mat');
        save(filename_simul, 'ModelData_simulation');
        fprintf('  ✓ ModelData_simulation saved: %s\n', filename_simul);
    end
    
    % Save ModelData_IRs (IRF data)
    if has_irfs
        filename_irs = fullfile(exp_paths.experiment, 'ModelData_IRs.mat');
        save(filename_irs, 'ModelData_IRs');
        fprintf('  ✓ ModelData_IRs saved: %s\n', filename_irs);
    end
else
    fprintf('\n  ⊘ Saving disabled (config.save_results = false)\n');
end

%% Nonlinearity and Preallocation Diagnostics
if has_1storder_simul || has_2ndorder_simul || has_pf_simul || has_irfs
    Diagnostics = print_nonlinearity_diagnostics(ModelData_simulation, AllShockResults, params, config, ModData);
    ModelData.Diagnostics = Diagnostics;
end

%% Summary Table (concise, copy-pasteable)
print_summary_table(config, params, calib_data, BaseResults, AllShockResults, ModelData, has_1storder_simul, has_2ndorder_simul, has_pf_simul, has_irfs, n_shocks, save_label);
