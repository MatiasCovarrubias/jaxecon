clear; clearvars -global; clc;

%% Paths
current_folder = fileparts(mfilename('fullpath'));
addpath(fullfile(current_folder, 'plotting'), '-begin');
addpath(fullfile(current_folder, 'utils'), '-begin');
addpath(fullfile(current_folder, 'dynare'), '-begin');
addpath(fullfile(current_folder, 'steady_state'), '-begin');
addpath(fullfile(current_folder, 'calibration'), '-begin');

%% Verify data files
required_files = {'calibration_data.mat', 'TFP_process.mat'};
missing_files = {};
for i = 1:numel(required_files)
    if ~exist(required_files{i}, 'file')
        missing_files{end+1} = required_files{i}; %#ok<SAGROW>
    end
end
if ~isempty(missing_files)
    error('main_IRs:MissingDataFiles', 'Missing: %s', strjoin(missing_files, ', '));
end

%% Configuration
config = struct();
config.model_type = 'VA';
config.save_results = false;
config.force_recalibrate = true;
config.sector_indices = [1];

config.date = "_Feb_2026";
config.exp_label = "_baseline";

config.gridpoints = 16;
config.sol_guess_file = 'SS_CDsolution_norm_permanent.mat';
config.fsolve_options = optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,...
    'MaxFunEvals',10000000,'MaxIter',10000);

config.run_firstorder_simul = true;
config.run_secondorder_simul = true;
config.run_pf_simul = true;
config.run_mit_shocks_simul = false;
config.run_firstorder_irs = true;
config.run_secondorder_irs = true;
config.run_pf_irs = true;

config.ir_horizon = 200;
config.ir_plot_length = 60;
config.plot_irs = false;

config.simul_T_firstorder = 1000;
config.simul_T_secondorder = 1000;
config.simul_T_pf = 1000;
config.pf_burn_in = 100;
config.pf_burn_out = 100;
config.simul_T_mit = 1000;
config.mit_burn_out = 100;

config.shock_sizes_pct = [20];

% Scale shock std. dev. for specific sectors: e.g., struct('sectors', [5 6], 'factor', 2.0)
config.shock_scaling = struct('sectors', [], 'factor', 1.0);

%% Parameters
params = struct();
params.beta = 0.96;
params.eps_l = 0.5;
params.eps_c = 0.3;
params.theta = 1;
params.phi = 4;
params.sigma_c = 0.5;
params.sigma_m = 0.001;
params.sigma_q = 0.5;
params.sigma_y = 0.6;
params.sigma_I = 0.5;
params.sigma_l = 0.1;

%% Derived settings
N_SECTORS = 37;
sector_indices = config.sector_indices;
validate_sector_indices(sector_indices, N_SECTORS, 'main_IRs');
save_label = strcat(config.date, config.exp_label);
config.shock_values = build_shock_values(config.shock_sizes_pct);

%% Experiment folder
exp_paths = setup_experiment_folder(save_label);
fprintf('\nExperiment: %s | Sectors: [%s]\n', save_label, num2str(sector_indices));

%% Load calibration
[calib_data, params] = load_calibration_data(params, sector_indices, config.model_type, config.shock_scaling);
labels = calib_data.labels;
fprintf('Calibration loaded (model: %s)\n', config.model_type);
print_empirical_targets(calib_data.empirical_targets);

%% Steady state
fprintf('\n--- Steady State ---\n');
ss_file = fullfile(exp_paths.experiment, 'steady_state.mat');
ss_cached = exist(ss_file, 'file') && ~config.force_recalibrate;

if ss_cached
    loaded = load(ss_file, 'ModData', 'params');
    ModData = loaded.ModData;
    params = loaded.params;
    fprintf('Loaded cached SS: %s\n', ss_file);
else
    fprintf('Targets: sig_c=%.2f sig_m=%.2f sig_q=%.2f sig_y=%.2f sig_I=%.2f sig_l=%.2f\n', ...
        params.sigma_c, params.sigma_m, params.sigma_q, params.sigma_y, params.sigma_I, params.sigma_l);

    calib_opts = struct();
    calib_opts.gridpoints = config.gridpoints;
    calib_opts.verbose = true;
    calib_opts.sol_guess_file = config.sol_guess_file;
    calib_opts.fsolve_options = config.fsolve_options;

    tic;
    [ModData, params] = calibrate_steady_state(params, calib_opts);
    elapsed_calib = toc;

    save(ss_file, 'ModData', 'params');
    fprintf('SS cached: %s (%.1fs)\n', ss_file, elapsed_calib);
end

%% Base simulation
fprintf('\n--- Dynare Analysis ---\n');
n_shocks = numel(config.shock_values);
run_any_irs = config.run_firstorder_irs || config.run_secondorder_irs || config.run_pf_irs;
run_any_simul = config.run_firstorder_simul || config.run_secondorder_simul || config.run_pf_simul || config.run_mit_shocks_simul;

AllShockResults = struct('DynareResults', {cell(n_shocks,1)}, 'IRFResults', {cell(n_shocks,1)});
BaseResults = struct();

if run_any_simul
    dynare_opts_base = build_dynare_opts(config, sector_indices, 'base');
    params.IRshock = config.shock_values(1).value;
    print_simulation_config(config);

    tic;
    BaseResults = run_dynare_analysis(ModData, params, dynare_opts_base);
    fprintf('Base simulation: %.1fs\n', toc);
end

%% IRF analysis
if run_any_irs
    [AllShockResults, BaseResults] = run_irf_loop(config, sector_indices, ...
        ModData, params, BaseResults, AllShockResults, exp_paths, labels);
else
    fprintf('\nIRF analysis skipped\n');
end

%% Package results
ModelData = build_ModelData(config, save_label, sector_indices, n_shocks, ...
    calib_data, labels, params, ModData, BaseResults, exp_paths);

fprintf('\n--- Simulation Results ---\n');
[ModelData_simulation, ModelData, flags] = build_ModelData_simulation( ...
    BaseResults, params, save_label, exp_paths, ModelData);

print_model_vs_empirical(BaseResults, calib_data);

if flags.has_pf
    fprintf('\n--- Capital Preallocation ---\n');
    prealloc_opts = struct('plot_figure', true, 'save_figure', config.save_results, ...
        'figures_folder', exp_paths.figures, 'save_label', save_label, ...
        'highlighted_sector', sector_indices(1));
    CapitalStats = analyze_capital_preallocation(BaseResults.SimulPerfectForesight, params, prealloc_opts, ModData);
    ModelData_simulation.PerfectForesight.CapitalStats = CapitalStats;
end

%% Build IRF data
has_irfs = run_any_irs && ~isempty(AllShockResults.IRFResults{1});
ModelData_IRs = build_ModelData_IRs(AllShockResults, config, save_label, sector_indices, n_shocks);
if has_irfs
    print_irf_summary(AllShockResults, config, n_shocks);
end

%% Save
save_experiment_results(config, exp_paths, ModelData, ModelData_simulation, ModelData_IRs, flags, has_irfs);

%% Diagnostics
if flags.has_1storder || flags.has_2ndorder || flags.has_pf || flags.has_mit || has_irfs
    Diagnostics = print_nonlinearity_diagnostics(ModelData_simulation, AllShockResults, params, config, ModData);
    ModelData.Diagnostics = Diagnostics;
end

%% Summary
print_summary_table(config, params, calib_data, BaseResults, AllShockResults, ModelData, ...
    flags.has_1storder, flags.has_2ndorder, flags.has_pf, flags.has_mit, has_irfs, n_shocks, save_label);
