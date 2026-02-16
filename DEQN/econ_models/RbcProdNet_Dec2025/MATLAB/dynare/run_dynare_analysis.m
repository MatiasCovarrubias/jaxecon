function Results = run_dynare_analysis(ModData, params, opts)

%% Defaults
if nargin < 3, opts = struct(); end

opts = set_default(opts, 'run_firstorder_simul', true);
opts = set_default(opts, 'run_secondorder_simul', false);
opts = set_default(opts, 'run_firstorder_irs', true);
opts = set_default(opts, 'run_secondorder_irs', false);
opts = set_default(opts, 'run_pf_irs', true);
opts = set_default(opts, 'run_pf_simul', true);
opts = set_default(opts, 'run_mit_shocks_simul', false);
opts = set_default(opts, 'sector_indices', [1]);
opts = set_default(opts, 'verbose', true);
opts = set_default(opts, 'ir_horizon', 200);
opts = set_default(opts, 'rng_seed', []);
opts = set_default(opts, 'simul_T_firstorder', 10000);
opts = set_default(opts, 'simul_T_secondorder', 10000);
opts = set_default(opts, 'simul_T_pf', 1000);
opts = set_default(opts, 'pf_burn_in', 100);
opts = set_default(opts, 'pf_burn_out', 100);
opts = set_default(opts, 'model_type', 'VA');
opts = set_default(opts, 'simul_T_mit', 200);
opts = set_default(opts, 'mit_burn_out', 50);

validate_params(params, {'n_sectors', 'IRshock', 'Sigma_A'}, 'run_dynare_analysis');
validate_sector_indices(opts.sector_indices, params.n_sectors, 'run_dynare_analysis');

n_sectors = params.n_sectors;
[dynare_folder, ~, ~] = fileparts(mfilename('fullpath'));
v = opts.verbose;

%% Generate shocks (only when at least one simulation is requested)
run_any_simul = opts.run_firstorder_simul || opts.run_secondorder_simul || ...
                opts.run_pf_simul || opts.run_mit_shocks_simul;

if run_any_simul
    T_candidates = [];
    if opts.run_firstorder_simul,  T_candidates(end+1) = opts.simul_T_firstorder; end
    if opts.run_secondorder_simul, T_candidates(end+1) = opts.simul_T_secondorder; end
    if opts.run_pf_simul,          T_candidates(end+1) = opts.simul_T_pf; end
    if opts.run_mit_shocks_simul,  T_candidates(end+1) = opts.simul_T_mit; end
    T_shocks = max(T_candidates);

    if ~isempty(opts.rng_seed), rng(opts.rng_seed); end
    rng_state = rng;
    shocks_master = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, T_shocks);

    Shocks = struct('data', shocks_master, 'T', T_shocks, ...
        'rng_state', rng_state, 'Sigma_A', params.Sigma_A);

    if v, fprintf('Shocks: %d x %d (seed=%d)\n', size(shocks_master,1), size(shocks_master,2), rng_state.Seed); end
else
    rng_state = rng;
    shocks_master = [];
    Shocks = struct('data', [], 'T', 0, 'rng_state', rng_state, 'Sigma_A', params.Sigma_A);
    if v, fprintf('IRF-only mode, skipping shock generation\n'); end
end

%% Prepare Dynare workspace
prepare_dynare_workspace(ModData, params, opts, dynare_folder);

idx = get_variable_indices(n_sectors);
policies_ss = ModData.policies_ss;
Cagg_ss = exp(policies_ss(idx.cagg - idx.ss_offset));
Lagg_ss = exp(policies_ss(idx.lagg - idx.ss_offset));

Results = struct('params', params, 'Cagg_ss', Cagg_ss, 'Lagg_ss', Lagg_ss);

%% 1. First-order solution (always required)
[oo_1st, M_1st, options_1st] = ensure_dynare_solution(1, dynare_folder, v);
Results.oo_1st = oo_1st;
Results.M_1st = M_1st;
Results.steady_state = oo_1st.steady_state;

%% Theoretical statistics
TheoStats = compute_theoretical_statistics(oo_1st, M_1st, policies_ss, n_sectors);
Results.TheoStats = TheoStats;

if v && ~isempty(fieldnames(TheoStats))
    fprintf('TheoStats: sig_GDP=%.4f sig_C=%.4f sig_I=%.4f sig_L=%.4f\n', ...
        TheoStats.sigma_VA_agg, TheoStats.sigma_C_agg, TheoStats.sigma_I_agg, TheoStats.sigma_L_agg);
end

%% 2. First-order simulation
if opts.run_firstorder_simul
    if v, fprintf('\n1st-order simulation (T=%d)...\n', opts.simul_T_firstorder); end
    tic;

    dynare_simul_1st = simult_(M_1st, options_1st, oo_1st.steady_state, oo_1st.dr, shocks_master(1:opts.simul_T_firstorder, :), 1);
    SolData = extract_state_space(oo_1st, M_1st, n_sectors, policies_ss);
    Shocks.usage.FirstOrder = struct('start', 1, 'end', opts.simul_T_firstorder);

    % Column 1 of simult_ output is the initial steady state; exclude it for statistics
    simul_data_1st = dynare_simul_1st(:, 2:end);
    variables_var = var(simul_data_1st, 0, 2);
    SolData.shocks_sd = sqrt(var(shocks_master, 0, 1)).';
    SolData.states_sd = sqrt(variables_var(1:idx.n_states));
    SolData.policies_sd = sqrt(variables_var(idx.n_states+1:idx.n_dynare));

    Results.SolData = SolData;
    Results.SimulFirstOrder = dynare_simul_1st;
    Results.ModelStats = compute_model_statistics(simul_data_1st, idx, policies_ss, n_sectors);

    if v, fprintf('1st-order done (%.1fs)\n', toc); end
end

%% 3. Second-order simulation
if opts.run_secondorder_simul
    [oo_2nd, M_2nd, options_2nd] = ensure_dynare_solution(2, dynare_folder, v);
    Results.oo_2nd = oo_2nd;
    Results.M_2nd = M_2nd;

    T2 = opts.simul_T_secondorder;
    if v, fprintf('\n2nd-order simulation (T=%d)...\n', T2); end
    tic;

    dynare_simul_2nd = simult_(M_2nd, options_2nd, oo_2nd.steady_state, oo_2nd.dr, shocks_master(1:T2,:), 2);
    Shocks.usage.SecondOrder = struct('start', 1, 'end', T2);

    Results.SimulSecondOrder = dynare_simul_2nd;
    % Column 1 of simult_ output is the initial steady state; exclude it for statistics
    simul_data_2nd = dynare_simul_2nd(:, 2:end);
    Results.ModelStats2nd = compute_model_statistics(simul_data_2nd, idx, policies_ss, n_sectors);

    if v, fprintf('2nd-order done (%.1fs)\n', toc); end
end

%% 4. First-order IRFs
if opts.run_firstorder_irs
    if v, fprintf('\n1st-order IRFs (H=%d)...\n', opts.ir_horizon); end
    tic;
    Results.IRSFirstOrder_raw = compute_perturbation_irfs( ...
        oo_1st, M_1st, options_1st, params, opts, 1);
    Results.ir_sector_indices = opts.sector_indices;
    if v, fprintf('1st-order IRFs done (%.1fs)\n', toc); end
end

%% 5. Second-order IRFs
if opts.run_secondorder_irs
    [oo_2nd, M_2nd, options_2nd] = ensure_dynare_solution(2, dynare_folder, v);
    if v, fprintf('\n2nd-order IRFs (H=%d)...\n', opts.ir_horizon); end
    tic;
    Results.IRSSecondOrder_raw = compute_perturbation_irfs( ...
        oo_2nd, M_2nd, options_2nd, params, opts, 2);
    if v, fprintf('2nd-order IRFs done (%.1fs)\n', toc); end
end

%% 6. Perfect foresight IRFs
if opts.run_pf_irs
    if v, fprintf('\nPF IRFs (H=%d)...\n', opts.ir_horizon); end
    tic;

    ir_simul_periods = opts.ir_horizon - 2;
    generate_determ_irs_mod(dynare_folder, ir_simul_periods);

    IRSDeterm_all = cell(numel(opts.sector_indices), 1);

    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        shocksim_0 = zeros([n_sectors, 1]);
        shocksim_0(sector_idx) = -params.IRshock;

        assignin('base', 'shocksim_0', shocksim_0);
        assignin('base', 'shockssim_ir', zeros([opts.ir_horizon, n_sectors]));

        run_dynare_mod(dynare_folder, 'determ_irs_generated');

        Simulated_time_series = evalin('base', 'Simulated_time_series');
        IRSDeterm_all{ii} = Simulated_time_series.data';
        if v, fprintf('  Sector %d done\n', sector_idx); end
    end

    Results.IRSPerfectForesight_raw = IRSDeterm_all;
    if v, fprintf('PF IRFs done (%.1fs)\n', toc); end
end

%% 7. Perfect foresight simulation
if opts.run_pf_simul
    burn_in = opts.pf_burn_in;
    burn_out = opts.pf_burn_out;
    T_active = opts.simul_T_pf;
    T_total = burn_in + T_active + burn_out;

    if v, fprintf('\nPF simulation (T_active=%d, burn_in=%d, burn_out=%d, total=%d)...\n', T_active, burn_in, burn_out, T_total); end
    tic;

    Shocks.usage.PerfectForesight = struct('start', 1, 'end', T_active);

    shockssim_pf = zeros(T_total, n_sectors);
    active_start = burn_in + 2;
    active_end = active_start + T_active - 1;
    shockssim_pf(active_start:active_end, :) = shocks_master(1:T_active, :);
    simul_periods = T_total - 2;
    shockssim_determ = shockssim_pf;
    assignin('base', 'shockssim_determ', shockssim_determ);

    generate_determ_simul_mod(dynare_folder, simul_periods);

    try
        run_dynare_mod(dynare_folder, 'determ_simul_generated');

        Simulated_time_series = evalin('base', 'Simulated_time_series');
        dynare_simul_pf_full = Simulated_time_series.data';

        expected_cols = T_total;
        actual_cols = size(dynare_simul_pf_full, 2);
        if actual_cols ~= expected_cols
            error('run_dynare_analysis:PFDimMismatch', ...
                'PF output has %d columns, expected %d (T_total=%d). Check periods= in determ_simul.mod', ...
                actual_cols, expected_cols, T_total);
        end

        dynare_simul_pf = dynare_simul_pf_full(:, burn_in+2 : burn_in+T_active+1);

        Results.SimulPerfectForesight = dynare_simul_pf;
        Results.SimulPerfectForesight_full = dynare_simul_pf_full;
        Results.pf_burn_in = burn_in;
        Results.pf_burn_out = burn_out;
        Results.pf_T_active = T_active;
        Results.pf_T_total = T_total;
        Results.ModelStatsPF = compute_model_statistics(dynare_simul_pf, idx, policies_ss, n_sectors);

        if v, fprintf('PF simulation done (%.1fs)\n', toc); end
    catch ME
        if v, fprintf('PF simulation failed: %s\n', ME.message); end
        Results.SimulPerfectForesight = [];
        Results.pf_simul_error = ME;
    end
end

%% 8. MIT shocks simulation
if opts.run_mit_shocks_simul
    T_mit = opts.simul_T_mit;
    burn_out_mit = opts.mit_burn_out;
    T_total_mit = T_mit + burn_out_mit;

    if v, fprintf('\nMIT shocks (T_active=%d, burn_out=%d, total=%d)...\n', T_mit, burn_out_mit, T_total_mit); end
    tic;

    shockssim_mit = shocks_master(1:T_mit, :);
    Shocks.usage.MITShocks = struct('start', 1, 'end', T_mit);
    assignin('base', 'shockssim_mit', shockssim_mit);
    generate_mit_shocks_mod(dynare_folder, T_mit, burn_out_mit);

    try
        run_dynare_mod(dynare_folder, 'mit_shocks_simul_generated');

        Simulated_time_series = evalin('base', 'Simulated_time_series');
        dynare_simul_mit_full = Simulated_time_series.data';

        expected_cols_mit = T_total_mit + 2;
        actual_cols_mit = size(dynare_simul_mit_full, 2);
        if actual_cols_mit ~= expected_cols_mit
            error('run_dynare_analysis:MITDimMismatch', ...
                'MIT output has %d columns, expected %d (periods=%d + 2). Check generate_mit_shocks_mod.', ...
                actual_cols_mit, expected_cols_mit, T_total_mit);
        end

        dynare_simul_mit = dynare_simul_mit_full(:, 2:T_mit+1);

        Results.SimulMITShocks = dynare_simul_mit;
        Results.SimulMITShocks_full = dynare_simul_mit_full;
        Results.mit_burn_in = 0;
        Results.mit_burn_out = burn_out_mit;
        Results.mit_T_active = T_mit;
        Results.mit_T_total = T_total_mit;
        Results.ModelStatsMIT = compute_model_statistics(dynare_simul_mit, idx, policies_ss, n_sectors);

        if v, fprintf('MIT shocks done (%.1fs)\n', toc); end
    catch ME
        if v, fprintf('MIT shocks failed: %s\n', ME.message); end
        Results.SimulMITShocks = [];
        Results.mit_simul_error = ME;
    end
end

%% Store shocks
Results.Shocks = Shocks;
Results.rng_state = rng_state;

if v, fprintf('\nDynare analysis complete. Fields: {%s}\n', strjoin(fieldnames(Results)', ', ')); end

end


%% ==================== Local Helpers ====================

function prepare_dynare_workspace(ModData, params, opts, dynare_folder)
    policies_ss = ModData.policies_ss;
    k_ss = ModData.endostates_ss;
    n_sectors = params.n_sectors;

    params_vars = struct2cell(ModData.parameters);
    params_names = fieldnames(ModData.parameters);
    for i = 1:numel(params_vars)
        assignin('base', params_names{i}, params_vars{i});
    end

    N = opts.ir_horizon;
    ax = 0:N-1;
    modstruct_path = fullfile(dynare_folder, 'ModStruct_temp.mat');
    save(modstruct_path, 'par*', 'policies_ss', 'k_ss', 'N', 'ax', '-regexp', '^par');

    model_config_path = fullfile(dynare_folder, 'model_config.mod');
    fid = fopen(model_config_path, 'w');
    fprintf(fid, '@#define n_sectors = %d\n', n_sectors);
    switch opts.model_type
        case 'VA',            fprintf(fid, '@#define MODEL_TYPE = 1\n');
        case {'GO', 'GO_noVA'}, fprintf(fid, '@#define MODEL_TYPE = 2\n');
    end
    fclose(fid);

    assignin('base', 'params', params);
    assignin('base', 'policies_ss', policies_ss);
    assignin('base', 'k_ss', k_ss);
    idx = get_variable_indices(n_sectors);
    assignin('base', 'Cagg_ss', exp(policies_ss(idx.cagg - idx.ss_offset)));
    assignin('base', 'Lagg_ss', exp(policies_ss(idx.lagg - idx.ss_offset)));
    assignin('base', 'parn_sectors', n_sectors);
end

function IRS_all = compute_perturbation_irfs(oo_, M_, options_, params, opts, order)
    n_sectors = params.n_sectors;
    IRS_all = cell(numel(opts.sector_indices), 1);

    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);

        ss_shocked = oo_.steady_state;
        ss_shocked(n_sectors + sector_idx) = -params.IRshock;

        shockssim_ir = zeros([opts.ir_horizon, n_sectors]);
        IRS_all{ii} = simult_(M_, options_, ss_shocked, oo_.dr, shockssim_ir, order);

        if opts.verbose, fprintf('  Sector %d done\n', sector_idx); end
    end
end
