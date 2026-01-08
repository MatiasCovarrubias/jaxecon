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
config.run_loglin_simul = true;    % Run log-linear (1st order) stochastic simulation
config.run_2ndorder_simul = true;  % Run 2nd order stochastic simulation
config.run_determ_simul = true;    % Run deterministic simulation (slower)
config.run_loglin_irs = true;      % Compute log-linear IRFs
config.run_determ_irs = true;      % Compute deterministic IRFs (slowest)
config.modorder = 1;               % Approximation order for log-linear simulation (keep at 1)

% --- IR settings ---
config.ir_horizon = 200;           % Horizon for IR calculation (needs to be long for convergence)
config.ir_plot_length = 60;        % Periods to plot in IR figures
config.plot_irs = true;            % Plot IRF figures (set false for batch runs)

% --- Simulation settings ---
config.simul_T_loglin = 2500;      % Log-linear (1st order) simulation length (fast)
config.simul_T_2ndorder = 2500;    % Second-order simulation length (fast)
config.simul_T_determ = 500;       % Deterministic simulation length (slower)

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
run_any_irs = config.run_loglin_irs || config.run_determ_irs;

% Storage for results
AllShockResults = struct();
AllShockResults.shock_configs = config.shock_values;
AllShockResults.n_shocks = n_shocks;
AllShockResults.DynareResults = cell(n_shocks, 1);
AllShockResults.IRFResults = cell(n_shocks, 1);
BaseResults = struct();

% Run base simulation (stoch_simul and optionally determ_simul)
if config.run_loglin_simul || config.run_2ndorder_simul || config.run_determ_simul
    dynare_opts_base = struct();
    dynare_opts_base.run_loglin_simul = config.run_loglin_simul;
    dynare_opts_base.run_2ndorder_simul = config.run_2ndorder_simul;
    dynare_opts_base.run_loglin_irs = false;
    dynare_opts_base.run_determ_irs = false;
    dynare_opts_base.run_determ_simul = config.run_determ_simul;
    dynare_opts_base.sector_indices = sector_indices;
    dynare_opts_base.modorder = config.modorder;
    dynare_opts_base.verbose = true;
    dynare_opts_base.ir_horizon = config.ir_horizon;
    dynare_opts_base.simul_T_loglin = config.simul_T_loglin;
    dynare_opts_base.simul_T_2ndorder = config.simul_T_2ndorder;
    dynare_opts_base.simul_T_determ = config.simul_T_determ;
    dynare_opts_base.model_type = config.model_type;
    
    params.IRshock = config.shock_values(1).value;
    
    fprintf('\n  ┌─ Base Simulation ─────────────────────────────────────────────┐\n');
    fprintf('  │  Configuration:                                               │\n');
    if config.run_loglin_simul
        fprintf('  │    • Log-linear (1st order):  ON  (T = %d)                 │\n', config.simul_T_loglin);
    else
        fprintf('  │    • Log-linear (1st order):  OFF                            │\n');
    end
    if config.run_2ndorder_simul
        fprintf('  │    • Second-order:            ON  (T = %d)                 │\n', config.simul_T_2ndorder);
    else
        fprintf('  │    • Second-order:            OFF                            │\n');
    end
    if config.run_determ_simul
        fprintf('  │    • Deterministic (PF):      ON  (T = %d)                  │\n', config.simul_T_determ);
    else
        fprintf('  │    • Deterministic (PF):      OFF                            │\n');
    end
    fprintf('  └────────────────────────────────────────────────────────────────┘\n\n');
    
    tic;
    BaseResults = run_dynare_analysis(ModData, params, dynare_opts_base);
    elapsed_base = toc;
    
    % Check what actually ran
    has_loglin = isfield(BaseResults, 'SimulLoglin') && ~isempty(BaseResults.SimulLoglin);
    has_2ndorder = isfield(BaseResults, 'Simul2ndOrder') && ~isempty(BaseResults.Simul2ndOrder);
    has_determ = isfield(BaseResults, 'SimulDeterm') && ~isempty(BaseResults.SimulDeterm);
    
    fprintf('\n  Base simulation results:\n');
    if has_loglin, fprintf('    ✓ SimulLoglin (1st order) computed\n'); else, fprintf('    ✗ SimulLoglin skipped\n'); end
    if has_2ndorder, fprintf('    ✓ Simul2ndOrder computed\n'); else, fprintf('    ✗ Simul2ndOrder skipped\n'); end
    if has_determ, fprintf('    ✓ SimulDeterm (PF) computed\n'); else, fprintf('    ✗ SimulDeterm skipped\n'); end
    if isfield(BaseResults, 'determ_simul_error')
        fprintf('    ⚠ Determ simulation failed: %s\n', BaseResults.determ_simul_error.message);
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
        dynare_opts.run_loglin_simul = false;
        dynare_opts.run_loglin_irs = config.run_loglin_irs;
        dynare_opts.run_determ_irs = config.run_determ_irs;
        dynare_opts.run_determ_simul = false;
        dynare_opts.sector_indices = sector_indices;
        dynare_opts.modorder = config.modorder;
        dynare_opts.verbose = true;
        dynare_opts.ir_horizon = config.ir_horizon;
        dynare_opts.simul_T_loglin = config.simul_T_loglin;
        dynare_opts.simul_T_determ = config.simul_T_determ;
        dynare_opts.model_type = config.model_type;
        
        tic;
        DynareResults = run_dynare_analysis(ModData, params, dynare_opts);
        elapsed_irf = toc;
        
        % Copy simulation data from base run (if available)
        if isfield(BaseResults, 'SolData')
            DynareResults.SolData = BaseResults.SolData;
        end
        if isfield(BaseResults, 'SimulLoglin')
            DynareResults.SimulLoglin = BaseResults.SimulLoglin;
        end
        if isfield(BaseResults, 'SimulDeterm')
            DynareResults.SimulDeterm = BaseResults.SimulDeterm;
        end
        if isfield(BaseResults, 'shockssim_determ')
            DynareResults.shockssim_determ = BaseResults.shockssim_determ;
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
ModelData_simulation = struct();
ModelData_simulation.metadata.save_label = save_label;
ModelData_simulation.metadata.exp_paths = exp_paths;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                       SIMULATION RESULTS                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════════════╝\n');

has_loglin_simul = false;
has_2ndorder_simul = false;
has_determ_simul = false;

% Log-linear simulation (if available)
if isfield(BaseResults, 'SimulLoglin') && ~isempty(BaseResults.SimulLoglin)
    has_loglin_simul = true;
    n = params.n_sectors;
    idx = get_variable_indices(n);
    Cagg_ss = BaseResults.Cagg_ss;
    Lagg_ss = BaseResults.Lagg_ss;
    
    ModelData_simulation.Loglin.full_simul = BaseResults.SimulLoglin;
    ModelData_simulation.Loglin.shocks = BaseResults.SolData.shockssim;
    ModelData_simulation.Loglin.variable_indices = idx;
    
    Cagg_loglin = exp(BaseResults.SimulLoglin(idx.cagg, :));
    Lagg_loglin = exp(BaseResults.SimulLoglin(idx.lagg, :));
    Yagg_loglin = exp(BaseResults.SimulLoglin(idx.yagg, :));
    Iagg_loglin = exp(BaseResults.SimulLoglin(idx.iagg, :));
    Magg_loglin = exp(BaseResults.SimulLoglin(idx.magg, :));
    
    ModelData_simulation.Loglin.Cagg = Cagg_loglin;
    ModelData_simulation.Loglin.Lagg = Lagg_loglin;
    ModelData_simulation.Loglin.Yagg = Yagg_loglin;
    ModelData_simulation.Loglin.Iagg = Iagg_loglin;
    ModelData_simulation.Loglin.Magg = Magg_loglin;
    
    ModelData_simulation.Loglin.Cagg_volatility = std(Cagg_loglin)/Cagg_ss;
    ModelData_simulation.Loglin.Lagg_volatility = std(Lagg_loglin)/Lagg_ss;
    
    % Also store volatilities in ModelData.Statistics for quick access
    ModelData.Statistics.Loglin.Cagg_volatility = ModelData_simulation.Loglin.Cagg_volatility;
    ModelData.Statistics.Loglin.Lagg_volatility = ModelData_simulation.Loglin.Lagg_volatility;
    
    fprintf('\n  Log-Linear Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', ...
        size(BaseResults.SimulLoglin, 1), size(BaseResults.SimulLoglin, 2));
    fprintf('    Cagg volatility: %.6f\n', ModelData_simulation.Loglin.Cagg_volatility);
    fprintf('    Lagg volatility: %.6f\n', ModelData_simulation.Loglin.Lagg_volatility);
    
    % Store simulation-based model statistics (for sectoral averages)
    if isfield(BaseResults, 'ModelStats')
        model_stats = BaseResults.ModelStats;
        ModelData_simulation.Loglin.ModelStats = model_stats;
        ModelData.Statistics.Loglin.ModelStats = model_stats;
    end
end

% Second-order simulation (if available)
if isfield(BaseResults, 'Simul2ndOrder') && ~isempty(BaseResults.Simul2ndOrder)
    has_2ndorder_simul = true;
    n = params.n_sectors;
    idx = get_variable_indices(n);
    Cagg_ss = BaseResults.Cagg_ss;
    Lagg_ss = BaseResults.Lagg_ss;
    Yagg_ss = ModData.Yagg_ss;
    Iagg_ss = ModData.Iagg_ss;
    Magg_ss = ModData.Magg_ss;
    
    ModelData_simulation.SecondOrder.full_simul = BaseResults.Simul2ndOrder;
    if isfield(BaseResults, 'shockssim_2nd')
        ModelData_simulation.SecondOrder.shocks = BaseResults.shockssim_2nd;
    end
    ModelData_simulation.SecondOrder.variable_indices = idx;
    
    Cagg_2nd = exp(BaseResults.Simul2ndOrder(idx.cagg, :));
    Lagg_2nd = exp(BaseResults.Simul2ndOrder(idx.lagg, :));
    Yagg_2nd = exp(BaseResults.Simul2ndOrder(idx.yagg, :));
    Iagg_2nd = exp(BaseResults.Simul2ndOrder(idx.iagg, :));
    Magg_2nd = exp(BaseResults.Simul2ndOrder(idx.magg, :));
    
    ModelData_simulation.SecondOrder.Cagg = Cagg_2nd;
    ModelData_simulation.SecondOrder.Lagg = Lagg_2nd;
    ModelData_simulation.SecondOrder.Yagg = Yagg_2nd;
    ModelData_simulation.SecondOrder.Iagg = Iagg_2nd;
    ModelData_simulation.SecondOrder.Magg = Magg_2nd;
    
    ModelData_simulation.SecondOrder.Cagg_volatility = std(Cagg_2nd)/Cagg_ss;
    ModelData_simulation.SecondOrder.Lagg_volatility = std(Lagg_2nd)/Lagg_ss;
    
    % Mean log deviations from SS (key for precautionary effects)
    ModelData_simulation.SecondOrder.Cagg_mean_logdev = mean(log(Cagg_2nd)) - log(Cagg_ss);
    ModelData_simulation.SecondOrder.Yagg_mean_logdev = mean(log(Yagg_2nd)) - log(Yagg_ss);
    ModelData_simulation.SecondOrder.Lagg_mean_logdev = mean(log(Lagg_2nd)) - log(Lagg_ss);
    ModelData_simulation.SecondOrder.Iagg_mean_logdev = mean(log(Iagg_2nd)) - log(Iagg_ss);
    ModelData_simulation.SecondOrder.Magg_mean_logdev = mean(log(Magg_2nd)) - log(Magg_ss);
    
    % Also store volatilities in ModelData.Statistics for quick access
    ModelData.Statistics.SecondOrder.Cagg_volatility = ModelData_simulation.SecondOrder.Cagg_volatility;
    ModelData.Statistics.SecondOrder.Lagg_volatility = ModelData_simulation.SecondOrder.Lagg_volatility;
    ModelData.Statistics.SecondOrder.Cagg_mean_logdev = ModelData_simulation.SecondOrder.Cagg_mean_logdev;
    ModelData.Statistics.SecondOrder.Yagg_mean_logdev = ModelData_simulation.SecondOrder.Yagg_mean_logdev;
    ModelData.Statistics.SecondOrder.Lagg_mean_logdev = ModelData_simulation.SecondOrder.Lagg_mean_logdev;
    ModelData.Statistics.SecondOrder.Iagg_mean_logdev = ModelData_simulation.SecondOrder.Iagg_mean_logdev;
    ModelData.Statistics.SecondOrder.Magg_mean_logdev = ModelData_simulation.SecondOrder.Magg_mean_logdev;
    
    fprintf('\n  Second-Order Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', ...
        size(BaseResults.Simul2ndOrder, 1), size(BaseResults.Simul2ndOrder, 2));
    fprintf('    Cagg volatility: %.6f\n', ModelData_simulation.SecondOrder.Cagg_volatility);
    fprintf('    Lagg volatility: %.6f\n', ModelData_simulation.SecondOrder.Lagg_volatility);
    
    % Store simulation-based model statistics (for sectoral averages)
    if isfield(BaseResults, 'ModelStats2nd')
        model_stats_2nd = BaseResults.ModelStats2nd;
        ModelData_simulation.SecondOrder.ModelStats = model_stats_2nd;
        ModelData.Statistics.SecondOrder.ModelStats = model_stats_2nd;
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

% Deterministic simulation (if available)
if isfield(BaseResults, 'SimulDeterm') && ~isempty(BaseResults.SimulDeterm)
    has_determ_simul = true;
    n = params.n_sectors;
    idx = get_variable_indices(n);
    Cagg_ss = BaseResults.Cagg_ss;
    Lagg_ss = BaseResults.Lagg_ss;
    Yagg_ss = ModData.Yagg_ss;
    Iagg_ss = ModData.Iagg_ss;
    Magg_ss = ModData.Magg_ss;
    
    ModelData_simulation.Determ.full_simul = BaseResults.SimulDeterm;
    if isfield(BaseResults, 'shockssim_determ')
        ModelData_simulation.Determ.shocks = BaseResults.shockssim_determ;
    end
    
    Cagg_determ = exp(BaseResults.SimulDeterm(idx.cagg, :));
    Lagg_determ = exp(BaseResults.SimulDeterm(idx.lagg, :));
    Yagg_determ = exp(BaseResults.SimulDeterm(idx.yagg, :));
    Iagg_determ = exp(BaseResults.SimulDeterm(idx.iagg, :));
    Magg_determ = exp(BaseResults.SimulDeterm(idx.magg, :));
    
    ModelData_simulation.Determ.Cagg = Cagg_determ;
    ModelData_simulation.Determ.Lagg = Lagg_determ;
    ModelData_simulation.Determ.Yagg = Yagg_determ;
    ModelData_simulation.Determ.Iagg = Iagg_determ;
    ModelData_simulation.Determ.Magg = Magg_determ;
    ModelData_simulation.Determ.Cagg_volatility = std(Cagg_determ)/Cagg_ss;
    ModelData_simulation.Determ.Lagg_volatility = std(Lagg_determ)/Lagg_ss;
    
    % Mean log deviations from SS (key for precautionary effects)
    ModelData_simulation.Determ.Cagg_mean_logdev = mean(log(Cagg_determ)) - log(Cagg_ss);
    ModelData_simulation.Determ.Yagg_mean_logdev = mean(log(Yagg_determ)) - log(Yagg_ss);
    ModelData_simulation.Determ.Lagg_mean_logdev = mean(log(Lagg_determ)) - log(Lagg_ss);
    ModelData_simulation.Determ.Iagg_mean_logdev = mean(log(Iagg_determ)) - log(Iagg_ss);
    ModelData_simulation.Determ.Magg_mean_logdev = mean(log(Magg_determ)) - log(Magg_ss);
    
    % Also store volatilities and mean log deviations in ModelData.Statistics for quick access
    ModelData.Statistics.Determ.Cagg_volatility = ModelData_simulation.Determ.Cagg_volatility;
    ModelData.Statistics.Determ.Lagg_volatility = ModelData_simulation.Determ.Lagg_volatility;
    ModelData.Statistics.Determ.Cagg_mean_logdev = ModelData_simulation.Determ.Cagg_mean_logdev;
    ModelData.Statistics.Determ.Yagg_mean_logdev = ModelData_simulation.Determ.Yagg_mean_logdev;
    ModelData.Statistics.Determ.Lagg_mean_logdev = ModelData_simulation.Determ.Lagg_mean_logdev;
    ModelData.Statistics.Determ.Iagg_mean_logdev = ModelData_simulation.Determ.Iagg_mean_logdev;
    ModelData.Statistics.Determ.Magg_mean_logdev = ModelData_simulation.Determ.Magg_mean_logdev;
    
    fprintf('\n  Deterministic (Perfect Foresight) Simulation:\n');
    fprintf('    Size: %d variables × %d periods\n', ...
        size(BaseResults.SimulDeterm, 1), size(BaseResults.SimulDeterm, 2));
    fprintf('    Cagg volatility: %.6f\n', ModelData_simulation.Determ.Cagg_volatility);
    fprintf('    Lagg volatility: %.6f\n', ModelData_simulation.Determ.Lagg_volatility);
    
    if has_loglin_simul
        fprintf('\n  ┌─ Volatility Comparison ─────────────────────────────────────┐\n');
        fprintf('  │                       Log-Linear     Deterministic           │\n');
        fprintf('  │  Cagg volatility:     %10.6f     %10.6f             │\n', ...
            ModelData_simulation.Loglin.Cagg_volatility, ...
            ModelData_simulation.Determ.Cagg_volatility);
        fprintf('  │  Lagg volatility:     %10.6f     %10.6f             │\n', ...
            ModelData_simulation.Loglin.Lagg_volatility, ...
            ModelData_simulation.Determ.Lagg_volatility);
        fprintf('  └────────────────────────────────────────────────────────────────┘\n');
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
    
    CapitalStats = analyze_capital_preallocation(BaseResults.SimulDeterm, params, prealloc_opts, ModData);
    ModelData_simulation.Determ.CapitalStats = CapitalStats;
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
    fprintf('  %-12s %6s %10s %10s %10s\n', 'Shock', 'A₀', 'Peak(LL)', 'Peak(NL)', 'Amplif');
    fprintf('  %s\n', repmat('─', 1, 52));
    for i = 1:n_shocks
        irf_res = AllShockResults.IRFResults{i};
        shock_cfg = config.shock_values(i);
        A_level = exp(-shock_cfg.value);  % Initial TFP level
        avg_peak_ll = mean(irf_res.Statistics.peak_values_loglin);
        avg_peak_det = mean(irf_res.Statistics.peak_values_determ);
        avg_amplif = mean(irf_res.Statistics.amplifications);
        fprintf('  %-12s %6.2f %10.4f %10.4f %10.4f\n', ...
            shock_cfg.label, A_level, avg_peak_ll, avg_peak_det, avg_amplif);
    end
    fprintf('  %s\n', repmat('─', 1, 52));
    fprintf('  Note: Peak = |max consumption deviation|, Amplif = Peak(NL) - Peak(LL)\n');
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
    if has_loglin_simul || has_2ndorder_simul || has_determ_simul
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
if has_loglin_simul || has_2ndorder_simul || has_determ_simul || has_irfs
    Diagnostics = print_nonlinearity_diagnostics(ModelData_simulation, AllShockResults, params, config, ModData);
    ModelData.Diagnostics = Diagnostics;
end

%% Summary Table (concise, copy-pasteable)
print_summary_table(config, params, calib_data, BaseResults, AllShockResults, ModelData, has_loglin_simul, has_2ndorder_simul, has_determ_simul, has_irfs, n_shocks, save_label);
