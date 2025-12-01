%%% This code runs the dynare codes for Carvalho, Covarrubias and Nuño (2022)
clear; 
clearvars -global;    
clc; 

%% Add paths (must be first to access utility functions)
addpath('calibration');
addpath('steady_state');
addpath('dynare');
addpath('utils');
addpath('plotting');

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

% --- General settings ---
config.save_results = false;       % Save data and graphs
config.recalibrate = true;         % Recalibrate steady state (false = load saved)
config.compute_all_sectors = false; % true = all 37 sectors, false = specified sectors
config.big_sector_ind = false;     % Only used when compute_all_sectors = false

% --- Experiment labels ---
config.date = "_December_2025";
config.exp_label = "_nonlinear_Min";

% --- Dynare analysis settings ---
config.run_loglin_simul = true;    % Run log-linear stochastic simulation
config.run_determ_simul = true;    % Run deterministic simulation (slower)
config.run_loglin_irs = true;      % Compute log-linear IRFs
config.run_determ_irs = true;      % Compute deterministic IRFs (slowest)
config.modorder = 1;               % Approximation order for stoch_simul

% --- IR settings ---
config.ir_horizon = 200;           % Horizon for IR calculation (needs to be long for convergence)
config.ir_plot_length = 60;        % Periods to plot in IR figures

% --- Simulation settings ---
config.simul_T_loglin = 10000;     % Log-linear simulation length (fast)
config.simul_T_determ = 1000;      % Deterministic simulation length (slower)

% --- Shock values configuration ---
% Each shock is defined as: {log_value, label, description}
% log_value: the shock to log(A), so log(0.8) = -0.2231 means A drops to 0.8
config.shock_values = [
    struct('value', -log(0.8), 'label', 'neg20pct', 'description', '-20% TFP (A=0.8)');
    struct('value', log(1.2),  'label', 'pos20pct', 'description', '+20% TFP (A=1.2)');
    struct('value', -log(0.95), 'label', 'neg5pct',  'description', '-5% TFP (A=0.95)');
    struct('value', log(1.05), 'label', 'pos5pct',  'description', '+5% TFP (A=1.05)')
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

%% Set up experiment folder structure
exp_paths = setup_experiment_folder(save_label);
disp(['Experiment: ' save_label]);
disp(['Experiment folder: ' exp_paths.experiment]);
disp(['Sectors: ' num2str(sector_indices)]);

%% Initialize params
params = struct();
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.33;
params.theta = 1;
params.phi = 4;

%% Load calibration data
[calib_data, params] = load_calibration_data(params, sector_indices);
labels = calib_data.labels;
disp('*** LOADED CALIBRATION DATA ***');

%% Steady State Calibration
if config.recalibrate
    params.sigma_c = config.sigma_c;
    params.sigma_m = config.sigma_m;
    params.sigma_q = config.sigma_q;
    params.sigma_y = config.sigma_y;
    params.sigma_I = config.sigma_I;
    params.sigma_l = config.sigma_l;
    
    calib_opts = struct();
    calib_opts.gridpoints = 8;
    calib_opts.verbose = true;
    calib_opts.sol_guess_file = 'SS_CDsolution_norm_permanent.mat';
    calib_opts.fsolve_options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,...
        'MaxFunEvals',10000000,'MaxIter',10000);
    
    tic;
    [ModData, params] = calibrate_steady_state(params, calib_opts);
    fprintf('Calibration time: %.2f seconds.\n', toc);
    
    ss_file = 'calibrated_steady_state.mat';
    save(ss_file, 'ModData', 'params');
    disp(['*** SAVED STEADY STATE: ' ss_file ' ***']);
else
    ss_file = 'calibrated_steady_state.mat';
    if exist(ss_file, 'file')
        load(ss_file, 'ModData', 'params');
        disp(['*** LOADED STEADY STATE: ' ss_file ' ***']);
    else
        error('main:MissingSteadyState', ...
            ['Steady state file not found: %s\n' ...
             'Set config.recalibrate = true to compute it.'], ss_file);
    end
end

%% Run Dynare Analysis
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
if config.run_loglin_simul || config.run_determ_simul
    dynare_opts_base = struct();
    dynare_opts_base.run_loglin_simul = config.run_loglin_simul;
    dynare_opts_base.run_loglin_irs = false;
    dynare_opts_base.run_determ_irs = false;
    dynare_opts_base.run_determ_simul = config.run_determ_simul;
    dynare_opts_base.sector_indices = sector_indices;
    dynare_opts_base.modorder = config.modorder;
    dynare_opts_base.verbose = true;
    dynare_opts_base.ir_horizon = config.ir_horizon;
    dynare_opts_base.simul_T_loglin = config.simul_T_loglin;
    dynare_opts_base.simul_T_determ = config.simul_T_determ;
    
    params.IRshock = config.shock_values(1).value;
    fprintf('\n--- Running base simulation ---\n');
    if config.run_loglin_simul
        fprintf('  - Log-linear stochastic simulation: ON (T=%d)\n', config.simul_T_loglin);
    end
    if config.run_determ_simul
        fprintf('  - Deterministic simulation: ON (T=%d)\n', config.simul_T_determ);
    end
    tic;
    BaseResults = run_dynare_analysis(ModData, params, dynare_opts_base);
    fprintf('Base simulation time: %.2f seconds.\n', toc);
    
    % Check what actually ran
    yesno = {'NO', 'YES'};
    has_loglin = isfield(BaseResults, 'SimulLoglin') && ~isempty(BaseResults.SimulLoglin);
    has_determ = isfield(BaseResults, 'SimulDeterm') && ~isempty(BaseResults.SimulDeterm);
    fprintf('\n--- Base simulation results ---\n');
    fprintf('  - SimulLoglin: %s\n', yesno{has_loglin + 1});
    fprintf('  - SimulDeterm: %s\n', yesno{has_determ + 1});
    if isfield(BaseResults, 'determ_simul_error')
        fprintf('  - WARNING: Determ simulation failed: %s\n', BaseResults.determ_simul_error.message);
    end
end

% Run IRF analysis for each shock value (if enabled)
if run_any_irs
    fprintf('\n=== RUNNING IRF ANALYSIS FOR %d SHOCK VALUES ===\n', n_shocks);
    
    for shock_idx = 1:n_shocks
        shock_config = config.shock_values(shock_idx);
        params.IRshock = shock_config.value;
        
        fprintf('\n=== SHOCK %d/%d: %s (value=%.4f) ===\n', ...
            shock_idx, n_shocks, shock_config.description, shock_config.value);
        
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
        
        tic;
        DynareResults = run_dynare_analysis(ModData, params, dynare_opts);
        fprintf('IRF computation time for shock %d: %.2f seconds.\n', shock_idx, toc);
        
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
        
        % Process IRFs for this shock
        ir_opts = struct();
        ir_opts.plot_graphs = ~config.compute_all_sectors && (shock_idx == 1);
        ir_opts.save_graphs = config.save_results;
        ir_opts.save_intermediate = config.save_results && config.compute_all_sectors;
        ir_opts.save_interval = 5;
        ir_opts.exp_paths = exp_paths;
        ir_opts.save_label = shock_config.label;
        ir_opts.ir_plot_length = config.ir_plot_length;
        
        IRFResults = process_sector_irs(DynareResults, params, ModData, labels, ir_opts);
        IRFResults.shock_config = shock_config;
        AllShockResults.IRFResults{shock_idx} = IRFResults;
    end
    
    fprintf('\n=== ALL SHOCKS PROCESSED ===\n');
else
    fprintf('\n=== IRF ANALYSIS SKIPPED (disabled in config) ===\n');
end

%% Build ModelData structure
ModelData = struct();

% Metadata
ModelData.metadata.date = config.date;
ModelData.metadata.exp_label = config.exp_label;
ModelData.metadata.save_label = save_label;
ModelData.metadata.sector_indices = sector_indices;
ModelData.metadata.sector_labels = labels.sector_labels;
ModelData.metadata.config = config;

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

% IRFs (if computed)
if run_any_irs && ~isempty(AllShockResults.IRFResults{1})
    ModelData.IRFs.by_shock = AllShockResults.IRFResults;
    ModelData.IRFs.shock_configs = config.shock_values;
    ModelData.IRFs.by_label = struct();
    for i = 1:n_shocks
        label = matlab.lang.makeValidName(config.shock_values(i).label);
        ModelData.IRFs.by_label.(label) = AllShockResults.IRFResults{i};
    end
end

% Simulation statistics (from base run, if available)
if isfield(BaseResults, 'SolData') && isfield(BaseResults.SolData, 'shocks_sd')
    ModelData.Simulation.shocks_sd = BaseResults.SolData.shocks_sd;
    ModelData.Simulation.states_sd = BaseResults.SolData.states_sd;
    ModelData.Simulation.policies_sd = BaseResults.SolData.policies_sd;
end

% RNG state for reproducibility
if isfield(BaseResults, 'rng_state')
    ModelData.Simulation.rng_state = BaseResults.rng_state;
end

%% Full simulation data (from base run, if available)
if isfield(BaseResults, 'SimulLoglin') && ~isempty(BaseResults.SimulLoglin)
    n = params.n_sectors;
    idx = get_variable_indices(n);
    Cagg_ss = BaseResults.Cagg_ss;
    Lagg_ss = BaseResults.Lagg_ss;
    
    ModelData.Simulation.Loglin.full_simul = BaseResults.SimulLoglin;
    ModelData.Simulation.Loglin.shocks = BaseResults.SolData.shockssim;
    ModelData.Simulation.Loglin.variable_indices = idx;
    
    Cagg_loglin = exp(BaseResults.SimulLoglin(idx.cagg, :));
    Lagg_loglin = exp(BaseResults.SimulLoglin(idx.lagg, :));
    Yagg_loglin = exp(BaseResults.SimulLoglin(idx.yagg, :));
    Iagg_loglin = exp(BaseResults.SimulLoglin(idx.iagg, :));
    Magg_loglin = exp(BaseResults.SimulLoglin(idx.magg, :));
    
    ModelData.Simulation.Loglin.Cagg = Cagg_loglin;
    ModelData.Simulation.Loglin.Lagg = Lagg_loglin;
    ModelData.Simulation.Loglin.Yagg = Yagg_loglin;
    ModelData.Simulation.Loglin.Iagg = Iagg_loglin;
    ModelData.Simulation.Loglin.Magg = Magg_loglin;
    
    ModelData.Simulation.Loglin.Cagg_volatility = std(Cagg_loglin)/Cagg_ss;
    ModelData.Simulation.Loglin.Lagg_volatility = std(Lagg_loglin)/Lagg_ss;
    
    fprintf('\n=== LOG-LINEAR SIMULATION STORED ===\n');
    fprintf('Simulation size: %d variables x %d periods\n', ...
        size(BaseResults.SimulLoglin, 1), size(BaseResults.SimulLoglin, 2));
    fprintf('Cagg volatility: %.4f\n', ModelData.Simulation.Loglin.Cagg_volatility);
    fprintf('Lagg volatility: %.4f\n', ModelData.Simulation.Loglin.Lagg_volatility);
end

% Deterministic simulation (if available)
if isfield(BaseResults, 'SimulDeterm') && ~isempty(BaseResults.SimulDeterm)
    n = params.n_sectors;
    idx = get_variable_indices(n);
    Cagg_ss = BaseResults.Cagg_ss;
    Lagg_ss = BaseResults.Lagg_ss;
    
    ModelData.Simulation.Determ.full_simul = BaseResults.SimulDeterm;
    if isfield(BaseResults, 'shockssim_determ')
        ModelData.Simulation.Determ.shocks = BaseResults.shockssim_determ;
    end
    
    Cagg_determ = exp(BaseResults.SimulDeterm(idx.cagg, :));
    Lagg_determ = exp(BaseResults.SimulDeterm(idx.lagg, :));
    
    ModelData.Simulation.Determ.Cagg = Cagg_determ;
    ModelData.Simulation.Determ.Lagg = Lagg_determ;
    ModelData.Simulation.Determ.Cagg_volatility = std(Cagg_determ)/Cagg_ss;
    ModelData.Simulation.Determ.Lagg_volatility = std(Lagg_determ)/Lagg_ss;
    
    fprintf('\n=== DETERMINISTIC SIMULATION STORED ===\n');
    fprintf('Simulation size: %d variables x %d periods\n', ...
        size(BaseResults.SimulDeterm, 1), size(BaseResults.SimulDeterm, 2));
    
    if isfield(ModelData.Simulation, 'Loglin')
        fprintf('\n=== VOLATILITY COMPARISON ===\n');
        fprintf('                     Log-Linear    Deterministic\n');
        fprintf('Cagg volatility:     %.4f        %.4f\n', ...
            ModelData.Simulation.Loglin.Cagg_volatility, ...
            ModelData.Simulation.Determ.Cagg_volatility);
        fprintf('Lagg volatility:     %.4f        %.4f\n', ...
            ModelData.Simulation.Loglin.Lagg_volatility, ...
            ModelData.Simulation.Determ.Lagg_volatility);
    end
end

%% Print IRF Summary (if IRFs were computed)
if run_any_irs && ~isempty(AllShockResults.IRFResults{1})
    fprintf('\n=== IRF SUMMARY ACROSS ALL SHOCKS ===\n');
    fprintf('%-15s %-12s %-12s %-12s\n', 'Shock', 'Peak(LL)', 'Peak(Det)', 'Amplif');
    fprintf('%s\n', repmat('-', 1, 55));
    for i = 1:n_shocks
        irf_res = AllShockResults.IRFResults{i};
        shock_cfg = config.shock_values(i);
        avg_peak_ll = mean(irf_res.Statistics.peak_values_loglin);
        avg_peak_det = mean(irf_res.Statistics.peak_values_determ);
        avg_amplif = mean(irf_res.Statistics.amplifications);
        fprintf('%-15s %-12.4f %-12.4f %-12.4f\n', ...
            shock_cfg.label, avg_peak_ll, avg_peak_det, avg_amplif);
    end
end

%% Save results and finalize
% Store experiment paths in ModelData for reference
ModelData.metadata.exp_paths = exp_paths;

if config.save_results
    filename = fullfile(exp_paths.experiment, 'ModelData.mat');
    save(filename, 'ModelData');
    disp(['*** SAVED: ' filename ' ***']);
end

%% Summary
fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Experiment folder: %s\n', exp_paths.experiment);
fprintf('\nModelData contents:\n');
fprintf('  - metadata, calibration, params: ALWAYS\n');
fprintf('  - SteadyState: ALWAYS\n');

yesno = {'NO', 'YES'};
has_solution = isfield(ModelData, 'Solution');
has_loglin = isfield(ModelData, 'Simulation') && isfield(ModelData.Simulation, 'Loglin');
has_determ = isfield(ModelData, 'Simulation') && isfield(ModelData.Simulation, 'Determ');
has_irfs = isfield(ModelData, 'IRFs');
fprintf('  - Solution: %s\n', yesno{has_solution + 1});
fprintf('  - Simulation.Loglin: %s\n', yesno{has_loglin + 1});
fprintf('  - Simulation.Determ: %s\n', yesno{has_determ + 1});
fprintf('  - IRFs: %s\n', yesno{has_irfs + 1});
