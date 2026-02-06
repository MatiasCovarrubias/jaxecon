function [Results] = run_dynare_analysis(ModData, params, opts)
% RUN_DYNARE_ANALYSIS Runs Dynare analysis including solution, simulation, and IRFs
%
% This function orchestrates all Dynare-based analysis with SEPARATE first-order
% and second-order perturbation solutions:
%
%   FIRST-ORDER (Linear Approximation):
%     1. Solves model with order=1 → computes ghx, ghu
%     2. Extracts A, B, C, D state-space matrices
%     3. Runs first-order simulation with random shocks
%     4. Computes first-order impulse responses
%
%   SECOND-ORDER (Quadratic Approximation):
%     5. Solves model with order=2 → computes ghx, ghu, ghxx, ghxu, ghuu, ghs2
%     6. Runs second-order simulation with pruning (captures precautionary effects)
%
%   PERFECT FORESIGHT (Nonlinear):
%     7. Perfect foresight IRFs (uses Newton solver, fully nonlinear)
%     8. Deterministic simulation with random shocks
%
% INPUTS:
%   ModData  - Structure from steady state calibration containing:
%              - parameters: calibrated parameters
%              - policies_ss: steady state policies
%              - endostates_ss: steady state capital (log)
%
%   params   - Structure with model parameters:
%              - n_sectors: Number of sectors
%              - IRshock: Impulse response shock value. Convention:
%                  TFP deviation = -IRshock, so A_0 = exp(-IRshock)
%                  For A to drop to A_neg: IRshock = -log(A_neg)
%                  For symmetric positive (A rises to 1/A_neg): IRshock = log(A_neg)
%              - Sigma_A: Covariance matrix of TFP shocks
%
%   opts     - Structure with analysis options:
%              - run_firstorder_simul: Run first-order simulation (default: true)
%              - run_secondorder_simul: Run second-order simulation (default: false)
%              - run_firstorder_irs: Run first-order IRFs (default: true)
%              - run_secondorder_irs: Run second-order IRFs (default: false)
%              - run_pf_irs: Run perfect foresight IRFs (default: true)
%              - run_pf_simul: Run perfect foresight simulation (default: true)
%              - sector_indices: Sectors to shock for IRFs (default: [1])
%              - verbose: Print progress (default: true)
%              - ir_horizon: Horizon for IR calculation (default: 200)
%              - simul_T_firstorder: First-order simulation length (default: 10000)
%              - simul_T_secondorder: Second-order simulation length (default: 10000)
%              - simul_T_pf: Perfect foresight simulation length (active periods with shocks, default: 500)
%              - pf_burn_in: Burn-in periods with no shocks before active simulation (default: 100)
%              - pf_burn_out: Burn-out periods with no shocks after active simulation (default: 100)
%              - rng_seed: RNG seed for reproducibility (default: [] = current state)
%
%   NOTE: Perfect foresight total periods = pf_burn_in + simul_T_pf + pf_burn_out.
%         Burn periods help convergence by letting model smoothly transition from/to SS.
%         IMPORTANT: The .mod file has hardcoded periods=698, which expects total=700.
%
%   NOTE: First-order solution (stoch_simul.mod) ALWAYS runs if not already computed.
%         Second-order solution (stoch_simul_2ndOrder.mod) runs if run_secondorder_simul=true OR run_secondorder_irs=true.
%
% OUTPUTS:
%   Results  - Structure with all analysis results:
%              - oo_1st, M_1st: First-order Dynare objects
%              - oo_2nd, M_2nd: Second-order Dynare objects (if computed)
%              - SolData: A, B, C, D state-space matrices (from first-order)
%              - SimulFirstOrder: First-order simulation results
%              - SimulSecondOrder: Second-order simulation results (if computed)
%              - IRSFirstOrder_raw: First-order impulse responses
%              - IRSSecondOrder_raw: Second-order impulse responses (if computed)
%              - IRSPerfectForesight_raw: Perfect foresight impulse responses

%% Input validation
validate_params(params, {'n_sectors', 'IRshock', 'Sigma_A'}, 'run_dynare_analysis');

%% Set default options
if nargin < 3
    opts = struct();
end

opts = set_default(opts, 'run_firstorder_simul', true);   % Run first-order (linear) simulation
opts = set_default(opts, 'run_secondorder_simul', false); % Run second-order (quadratic) simulation
opts = set_default(opts, 'run_firstorder_irs', true);     % Run first-order IRFs
opts = set_default(opts, 'run_secondorder_irs', false);   % Run second-order IRFs
opts = set_default(opts, 'run_pf_irs', true);             % Run perfect foresight IRFs
opts = set_default(opts, 'run_pf_simul', true);           % Run perfect foresight simulation
opts = set_default(opts, 'sector_indices', [1]);
opts = set_default(opts, 'verbose', true);
opts = set_default(opts, 'ir_horizon', 200);  % IR calculation horizon
opts = set_default(opts, 'rng_seed', []);  % Empty = use current RNG state; integer = set specific seed
opts = set_default(opts, 'simul_T_firstorder', 10000);   % First-order simulation length
opts = set_default(opts, 'simul_T_secondorder', 10000);  % Second-order simulation length
opts = set_default(opts, 'simul_T_pf', 500);             % Perfect foresight: active periods with shocks
opts = set_default(opts, 'pf_burn_in', 100);             % Burn-in periods (no shocks) before active
opts = set_default(opts, 'pf_burn_out', 100);            % Burn-out periods (no shocks) after active
opts = set_default(opts, 'model_type', 'VA');  % Model type: 'VA', 'GO', or 'GO_noVA'

% Validate sector indices
validate_sector_indices(opts.sector_indices, params.n_sectors, 'run_dynare_analysis');

n_sectors = params.n_sectors;
simul_T_firstorder = opts.simul_T_firstorder;
simul_T_pf = opts.simul_T_pf;

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

% Generate model_config.mod file for Dynare preprocessor
% This sets the model type (VA vs GO) as a Dynare macro variable
model_config_path = fullfile(dynare_folder, 'model_config.mod');
fid = fopen(model_config_path, 'w');
switch opts.model_type
    case 'VA'
        fprintf(fid, '// Model configuration: Value Added (TFP multiplies Y)\n');
        fprintf(fid, '@#define MODEL_TYPE = 1\n');
        fprintf(fid, '// MODEL_TYPE = 1: VA (TFP in Y equation)\n');
        fprintf(fid, '// MODEL_TYPE = 2: GO (TFP in Q equation)\n');
    case {'GO', 'GO_noVA'}
        fprintf(fid, '// Model configuration: Gross Output (TFP multiplies Q)\n');
        fprintf(fid, '@#define MODEL_TYPE = 2\n');
        fprintf(fid, '// MODEL_TYPE = 1: VA (TFP in Y equation)\n');
        fprintf(fid, '// MODEL_TYPE = 2: GO (TFP in Q equation)\n');
end
fclose(fid);
if opts.verbose
    fprintf('     Model type: %s\n', opts.model_type);
end

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

%% 1. First-Order Solution (Linear Approximation) - ALWAYS REQUIRED
% This computes ghx, ghu for the A, B, C, D state-space matrices
have_saved_1st = false;
try
    oo_1st = evalin('base', 'oo_1st_');
    if isstruct(oo_1st.dr) && isfield(oo_1st.dr, 'ghx')
        have_saved_1st = true;
    end
catch
    % No saved first-order objects
end

if ~have_saved_1st
    if opts.verbose
        fprintf('\n  ── First-Order Solution (stoch_simul, order=1) ─────────────────\n');
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
        fprintf('     ✓ First-order solution completed (%.2f s)\n', elapsed);
    end
    
    % Save first-order objects (deterministic/2nd-order will overwrite oo_, M_, options_)
    oo_1st = evalin('base', 'oo_');
    M_1st = evalin('base', 'M_');
    options_1st = evalin('base', 'options_');
    assignin('base', 'oo_1st_', oo_1st);
    assignin('base', 'M_1st_', M_1st);
    assignin('base', 'options_1st_', options_1st);
    
    if opts.verbose
        fprintf('     ✓ First-order objects cached (ghx, ghu)\n');
    end
else
    if opts.verbose
        fprintf('\n  ✓ Using cached first-order solution\n');
    end
end

% Get Dynare objects from saved first-order solution
oo_ = evalin('base', 'oo_1st_');
M_ = evalin('base', 'M_1st_');
options_ = evalin('base', 'options_1st_');

Results.oo_1st = oo_;
Results.M_1st = M_;
Results.steady_state = oo_.steady_state;

%% Extract Theoretical Statistics from State Space
% These are model-implied moments computed analytically from the solution,
% not from simulation. Available immediately after stoch_simul with periods=0.
TheoStats = compute_theoretical_statistics(oo_, M_, policies_ss, n_sectors);
Results.TheoStats = TheoStats;

if opts.verbose && ~isempty(fieldnames(TheoStats))
    fprintf('\n  ── Theoretical Moments (from state-space solution) ─────────────\n');
    fprintf('     σ(Y_agg):  %.4f   (aggregate GDP)\n', TheoStats.sigma_VA_agg);
    fprintf('     σ(C_agg):  %.4f   (aggregate consumption)\n', TheoStats.sigma_C_agg);
    fprintf('     σ(L_agg):  %.4f   (aggregate labor)\n', TheoStats.sigma_L_agg);
    fprintf('     σ(I_agg):  %.4f   (aggregate investment)\n', TheoStats.sigma_I_agg);
    fprintf('     σ(M_agg):  %.4f   (aggregate intermediates)\n', TheoStats.sigma_M_agg);
    if isfield(TheoStats, 'rho_VA_agg')
        fprintf('     ρ(Y_agg):  %.4f   (GDP autocorrelation)\n', TheoStats.rho_VA_agg);
    end
end

%% 2. First-Order Simulation (with random shocks)
if opts.run_firstorder_simul
    if opts.verbose
        fprintf('\n  ── First-Order Simulation (T = %d) ───────────────────────────\n', simul_T_firstorder);
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
    shockssim = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, simul_T_firstorder);
    
    % Simulate using Dynare's simult_ with first-order solution (order=1)
    % Uses only ghx, ghu (linear policy functions)
    dynare_simul_1st = simult_(M_, options_, oo_.steady_state, oo_.dr, shockssim, 1);
    
    % Extract solution matrices (State Space Representation)
    % S(t) = A*S(t-1) + B*e(t)
    % X(t) = C*S(t-1) + D*e(t)
    SolData = extract_state_space(oo_, M_, n_sectors, policies_ss);
    SolData.shockssim = shockssim;
    
    % Compute simulation statistics
    varlev = exp(dynare_simul_1st(1:idx.n_dynare,:));
    variables_var = var(dynare_simul_1st, 0, 2);
    
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
    Results.SimulFirstOrder = dynare_simul_1st;
    
    % Compute model statistics (comparable to HP-filtered empirical targets)
    ModelStats = compute_model_statistics(dynare_simul_1st, idx, policies_ss, n_sectors);
    Results.ModelStats = ModelStats;
    
    if opts.verbose
        fprintf('\n     Model Statistics (first-order simulation):\n');
        fprintf('       σ(Y_agg):  %.4f\n', ModelStats.sigma_VA_agg);
        fprintf('       σ(L) avg (VA-wgt):  %.4f   | σ(L) avg (emp-wgt):  %.4f\n', ...
            ModelStats.sigma_L_avg, ModelStats.sigma_L_avg_empweighted);
        fprintf('       σ(I) avg (VA-wgt):  %.4f   | σ(I) avg (inv-wgt):  %.4f\n', ...
            ModelStats.sigma_I_avg, ModelStats.sigma_I_avg_invweighted);
    end
end

%% 2b. Second-Order Solution and Simulation
if opts.run_secondorder_simul
    simul_T_secondorder = opts.simul_T_secondorder;
    
    % First, compute second-order solution (separate from first-order)
    have_saved_2nd = false;
    try
        oo_2nd = evalin('base', 'oo_2nd_');
        if isstruct(oo_2nd.dr) && isfield(oo_2nd.dr, 'ghxx')
            have_saved_2nd = true;
        end
    catch
        % No saved second-order objects
    end
    
    if ~have_saved_2nd
        if opts.verbose
            fprintf('\n  ── Second-Order Solution (stoch_simul_2ndOrder, order=2) ──────\n');
        end
        
        tic;
        
        % Change to dynare folder, run 2nd order, and return
        current_dir = pwd;
        cd(dynare_folder);
        try
            dynare stoch_simul_2ndOrder;
        catch ME
            cd(current_dir);
            rethrow(ME);
        end
        cd(current_dir);
        
        elapsed = toc;
        if opts.verbose
            fprintf('     ✓ Second-order solution completed (%.2f s)\n', elapsed);
        end
        
        % Save second-order objects
        oo_2nd = evalin('base', 'oo_');
        M_2nd = evalin('base', 'M_');
        options_2nd = evalin('base', 'options_');
        assignin('base', 'oo_2nd_', oo_2nd);
        assignin('base', 'M_2nd_', M_2nd);
        assignin('base', 'options_2nd_', options_2nd);
        
        if opts.verbose
            fprintf('     ✓ Second-order objects cached (ghx, ghu, ghxx, ghxu, ghuu, ghs2)\n');
            fprintf('     ⚠ No pruning: uses full quadratic dynamics\n');
        end
    else
        if opts.verbose
            fprintf('\n  ✓ Using cached second-order solution\n');
        end
        oo_2nd = evalin('base', 'oo_2nd_');
        M_2nd = evalin('base', 'M_2nd_');
        options_2nd = evalin('base', 'options_2nd_');
    end
    
    Results.oo_2nd = oo_2nd;
    Results.M_2nd = M_2nd;
    
    % Now run second-order simulation
    if opts.verbose
        fprintf('\n  ── Second-Order Simulation (T = %d) ─────────────────────────\n', simul_T_secondorder);
    end
    
    tic;
    
    % Use same RNG state as first-order for comparability (if available)
    if isfield(Results, 'rng_state')
        rng(Results.rng_state);
        if opts.verbose
            fprintf('     RNG: Using same seed as first-order simulation\n');
        end
    elseif ~isempty(opts.rng_seed)
        if isstruct(opts.rng_seed)
            rng(opts.rng_seed);
        else
            rng(opts.rng_seed);
        end
    end
    
    % Generate random shocks (same as first-order if same RNG state)
    shockssim_2nd = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, simul_T_secondorder);
    
    % Simulate using Dynare's simult_ with order=2
    % Uses ghx, ghu (linear) + ghxx, ghxu, ghuu, ghs2 (quadratic terms)
    dynare_simul_2nd = simult_(M_2nd, options_2nd, oo_2nd.steady_state, oo_2nd.dr, shockssim_2nd, 2);
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed (%.2f s)\n', elapsed);
    end
    
    Results.SimulSecondOrder = dynare_simul_2nd;
    Results.shockssim_2nd = shockssim_2nd;
    
    % Compute model statistics for 2nd order
    ModelStats2nd = compute_model_statistics(dynare_simul_2nd, idx, policies_ss, n_sectors);
    Results.ModelStats2nd = ModelStats2nd;
    
    if opts.verbose
        fprintf('\n     Model Statistics (2nd-order simulation):\n');
        fprintf('       σ(Y_agg):  %.4f\n', ModelStats2nd.sigma_VA_agg);
        fprintf('       σ(L) avg (VA-wgt):  %.4f   | σ(L) avg (emp-wgt):  %.4f\n', ...
            ModelStats2nd.sigma_L_avg, ModelStats2nd.sigma_L_avg_empweighted);
        fprintf('       σ(I) avg (VA-wgt):  %.4f   | σ(I) avg (inv-wgt):  %.4f\n', ...
            ModelStats2nd.sigma_I_avg, ModelStats2nd.sigma_I_avg_invweighted);
    end
end

%% 3. First-Order Impulse Responses
if opts.run_firstorder_irs
    if opts.verbose
        fprintf('\n  ── First-Order IRFs (horizon = %d) ──────────────────────────\n', opts.ir_horizon);
    end
    
    tic;
    IRSLoglin_all = cell(numel(opts.sector_indices), 1);
    
    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        
        % Set shock in initial TFP (log deviation from steady state)
        % Convention: TFP deviation = -params.IRshock
        %   - Negative shock (A drops to A_neg): IRshock = -log(A_neg)
        %   - Positive shock (A rises to A_pos): IRshock = -log(A_pos) = log(1/A_pos)
        %   - Symmetric pair: IRshock_pos = -IRshock_neg (same |shock| in log space)
        steady_state_shocked = oo_.steady_state;
        steady_state_shocked(n_sectors + sector_idx) = -params.IRshock;
        
        % Zero exogenous shocks (shock is in initial condition)
        shockssim_ir = zeros([opts.ir_horizon, n_sectors]);
        
        % Simulate using first-order solution (order=1)
        dynare_simul = simult_(M_, options_, steady_state_shocked, oo_.dr, shockssim_ir, 1);
        
        IRSLoglin_all{ii} = dynare_simul;
        
        if opts.verbose
            fprintf('     • Sector %d\n', sector_idx);
        end
    end
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed %d sectors (%.2f s)\n', numel(opts.sector_indices), elapsed);
    end
    
    Results.IRSFirstOrder_raw = IRSLoglin_all;
    Results.ir_sector_indices = opts.sector_indices;
end

%% 3b. Second-Order Impulse Responses
if opts.run_secondorder_irs
    % Ensure second-order solution is available
    have_saved_2nd = false;
    try
        oo_2nd = evalin('base', 'oo_2nd_');
        if isstruct(oo_2nd.dr) && isfield(oo_2nd.dr, 'ghxx')
            have_saved_2nd = true;
        end
    catch
        % No saved second-order objects
    end
    
    if ~have_saved_2nd
        if opts.verbose
            fprintf('\n  ── Second-Order Solution (required for 2nd-order IRFs) ────────\n');
        end
        
        tic;
        current_dir = pwd;
        cd(dynare_folder);
        try
            dynare stoch_simul_2ndOrder;
        catch ME
            cd(current_dir);
            rethrow(ME);
        end
        cd(current_dir);
        
        elapsed = toc;
        if opts.verbose
            fprintf('     ✓ Second-order solution completed (%.2f s)\n', elapsed);
        end
        
        oo_2nd = evalin('base', 'oo_');
        M_2nd = evalin('base', 'M_');
        options_2nd = evalin('base', 'options_');
        assignin('base', 'oo_2nd_', oo_2nd);
        assignin('base', 'M_2nd_', M_2nd);
        assignin('base', 'options_2nd_', options_2nd);
    else
        oo_2nd = evalin('base', 'oo_2nd_');
        M_2nd = evalin('base', 'M_2nd_');
        options_2nd = evalin('base', 'options_2nd_');
    end
    
    if opts.verbose
        fprintf('\n  ── Second-Order IRFs (horizon = %d) ─────────────────────────\n', opts.ir_horizon);
    end
    
    tic;
    IRS2ndOrder_all = cell(numel(opts.sector_indices), 1);
    
    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        
        % Set shock in initial TFP (log deviation from steady state)
        % Same convention as first-order IRFs
        steady_state_shocked = oo_2nd.steady_state;
        steady_state_shocked(n_sectors + sector_idx) = -params.IRshock;
        
        % Zero exogenous shocks (shock is in initial condition)
        shockssim_ir = zeros([opts.ir_horizon, n_sectors]);
        
        % Simulate using second-order solution (order=2, no pruning)
        dynare_simul = simult_(M_2nd, options_2nd, steady_state_shocked, oo_2nd.dr, shockssim_ir, 2);
        
        IRS2ndOrder_all{ii} = dynare_simul;
        
        if opts.verbose
            fprintf('     • Sector %d\n', sector_idx);
        end
    end
    
    elapsed = toc;
    if opts.verbose
        fprintf('     ✓ Completed %d sectors (%.2f s)\n', numel(opts.sector_indices), elapsed);
    end
    
    Results.IRSSecondOrder_raw = IRS2ndOrder_all;
end

%% 4. Perfect Foresight Impulse Responses
if opts.run_pf_irs
    if opts.verbose
        fprintf('\n  ── Perfect Foresight IRFs (horizon = %d) ──────────────────────\n', opts.ir_horizon);
    end
    
    tic;
    IRSDeterm_all = cell(numel(opts.sector_indices), 1);
    
    for ii = 1:numel(opts.sector_indices)
        sector_idx = opts.sector_indices(ii);
        
        % Set initial shock (same convention as first-order IRFs)
        % TFP deviation = -params.IRshock
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
    
    Results.IRSPerfectForesight_raw = IRSDeterm_all;
end

%% 5. Perfect Foresight Simulation (with Random Shocks)
if opts.run_pf_simul
    % Calculate total simulation length with burn-in/burn-out periods
    burn_in = opts.pf_burn_in;
    burn_out = opts.pf_burn_out;
    T_active = simul_T_pf;  % Periods with shocks
    T_total = burn_in + T_active + burn_out;  % Total periods
    
    if opts.verbose
        fprintf('\n  ── Perfect Foresight Simulation ─────────────────────────────\n');
        fprintf('     Active periods (with shocks): %d\n', T_active);
        fprintf('     Burn-in (no shocks): %d | Burn-out (no shocks): %d\n', burn_in, burn_out);
        fprintf('     Total simulation: %d periods\n', T_total);
    end
    
    tic;
    
    % Generate random shocks for the ACTIVE period only
    shockssim_active = mvnrnd(zeros([n_sectors,1]), params.Sigma_A, T_active);
    
    % Build full shock matrix with burn-in/burn-out (zeros at boundaries)
    % Dynare expects (T_total x n_exo): [initial; simulation periods; terminal]
    % Row 1: initial period (t=0)
    % Rows 2:T_total-1: simulation periods (t=1 to T_total-2)
    % Row T_total: terminal period (t=T_total-1)
    %
    % Shock placement:
    %   - Rows 1 to burn_in+1: zeros (burn-in + initial)
    %   - Rows burn_in+2 to burn_in+1+T_active: active shocks
    %   - Rows burn_in+2+T_active to T_total: zeros (burn-out + terminal)
    
    shockssim_pf = zeros(T_total, n_sectors);
    
    % Place active shocks after burn-in period
    active_start = burn_in + 2;  % First row for active shocks (after initial + burn-in)
    active_end = active_start + T_active - 1;
    shockssim_pf(active_start:active_end, :) = shockssim_active;
    
    % Dynare periods = T_total - 2 (excluding initial and terminal)
    simul_periods = T_total - 2;
    
    if opts.verbose
        fprintf('     Shock matrix: %d × %d | Dynare periods: %d\n', ...
            size(shockssim_pf, 1), size(shockssim_pf, 2), simul_periods);
        fprintf('     Active shock rows: %d to %d\n', active_start, active_end);
    end
    
    % Verify the hardcoded periods in determ_simul.mod matches
    % Expected: periods = T_total - 2 = 698 for T_total = 700
    expected_mod_periods = T_total - 2;
    if expected_mod_periods ~= 698
        warning(['Perfect foresight expects periods=%d, but determ_simul.mod has periods=698 hardcoded.\n' ...
                 'Update determ_simul.mod to match, or adjust burn_in/burn_out/simul_T_pf to get T_total=700.'], ...
                expected_mod_periods);
    end
    
    % Save to workspace (needed by .mod file)
    % NOTE: Using 'shockssim_determ' to match determ_simul.mod variable name
    shockssim_determ = shockssim_pf;
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
        dynare_simul_pf_full = Simulated_time_series.data';  % Full simulation including burn periods
        
        % Extract only the active periods (excluding burn-in and burn-out)
        % Simulation output has T_total rows, we want rows burn_in+1 to burn_in+T_active
        active_simul_start = burn_in + 1;
        active_simul_end = burn_in + T_active;
        dynare_simul_pf = dynare_simul_pf_full(:, active_simul_start:active_simul_end);
        
        if opts.verbose
            fprintf('     Full simulation: %d periods | Active extracted: %d periods\n', ...
                size(dynare_simul_pf_full, 2), size(dynare_simul_pf, 2));
        end
        
        % Store results
        Results.SimulPerfectForesight = dynare_simul_pf;  % Active periods only
        Results.SimulPerfectForesight_full = dynare_simul_pf_full;  % Full including burn
        Results.shockssim_pf = shockssim_active;  % Only active shocks
        Results.shockssim_pf_full = shockssim_pf;  % Full shock matrix
        Results.pf_burn_in = burn_in;
        Results.pf_burn_out = burn_out;
        Results.pf_T_active = T_active;
        Results.pf_T_total = T_total;
        
        % Compute model statistics for perfect foresight (comparable to 1st/2nd order)
        ModelStatsPF = compute_model_statistics(dynare_simul_pf, idx, policies_ss, n_sectors);
        Results.ModelStatsPF = ModelStatsPF;
        
        elapsed = toc;
        if opts.verbose
            fprintf('     ✓ Completed (%.2f s)\n', elapsed);
            fprintf('\n     Model Statistics (perfect foresight simulation):\n');
            fprintf('       σ(Y_agg):  %.4f\n', ModelStatsPF.sigma_VA_agg);
            fprintf('       σ(L) avg (VA-wgt):  %.4f   | σ(L) avg (emp-wgt):  %.4f\n', ...
                ModelStatsPF.sigma_L_avg, ModelStatsPF.sigma_L_avg_empweighted);
            fprintf('       σ(I) avg (VA-wgt):  %.4f   | σ(I) avg (inv-wgt):  %.4f\n', ...
                ModelStatsPF.sigma_I_avg, ModelStatsPF.sigma_I_avg_invweighted);
        end
    catch ME
        cd(current_dir);
        if opts.verbose
            fprintf('     ⚠ Failed: %s\n', ME.message);
        end
        Results.SimulPerfectForesight = [];
        Results.pf_simul_error = ME;
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
%     - sigma_VA_agg: Aggregate GDP volatility (from model's yagg)
%     - sigma_L_agg: Aggregate labor volatility (from model's lagg, CES aggregator)
%     - sigma_L_hc_agg: Aggregate labor volatility (headcount = simple sum, comparable to data)
%     - sigma_I_agg: Aggregate investment volatility (from model's iagg)
%     - sigma_M_agg: Aggregate intermediates volatility (from model's magg)
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
    sigma_L_agg = std(lagg_simul);  % From model's CES aggregator (preferences-based)
    sigma_I_agg = std(iagg_simul);
    sigma_M_agg = std(magg_simul);
    
    % Compute headcount labor aggregate (simple sum, comparable to data)
    % l_simul is in log deviations, so exp(l_simul) gives L_j(t)/L_j^ss
    l_ss_idx = (idx.l(1):idx.l(2)) - idx.ss_offset;
    l_ss_log = policies_ss(l_ss_idx);
    l_ss = exp(l_ss_log);  % Steady state labor by sector
    
    % L_hc(t) = sum_j L_j(t) = sum_j L_j^ss * exp(l_simul(j,t))
    L_hc_ss = sum(l_ss);
    L_hc_levels = l_ss' * exp(l_simul);  % Sum of sectoral labor in levels
    lagg_hc_simul = log(L_hc_levels) - log(L_hc_ss);  % Log deviation from SS
    sigma_L_hc_agg = std(lagg_hc_simul);
    
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
    
    %% ===== OWN-VARIABLE WEIGHTED SECTORAL VOLATILITIES =====
    % These match the empirical targets: employment-weighted labor vol, investment-weighted inv vol
    % Data uses time-average shares; model analog is steady-state shares (SS = long-run average)
    
    % Compute employment weights from steady state (L_j^ss / sum L_j^ss)
    % Data: time-avg of employment shares (headcount, real units)
    emp_weights = l_ss' / sum(l_ss);
    
    % Compute nominal investment weights from steady state
    % Data uses NOMINAL investment shares: Invn / sum(Invn)
    % Model analog: (I_j * P_k_j) / sum(I_j * P_k_j) at steady state
    i_ss_idx = (idx.i(1):idx.i(2)) - idx.ss_offset;
    i_ss_log = policies_ss(i_ss_idx);
    i_ss = exp(i_ss_log);  % Steady state real investment by sector
    
    pk_ss_idx = (idx.pk(1):idx.pk(2)) - idx.ss_offset;
    pk_ss_log = policies_ss(pk_ss_idx);
    pk_ss = exp(pk_ss_log);  % Steady state investment price by sector
    
    i_nominal_ss = i_ss .* pk_ss;  % Nominal investment = I * P_k
    inv_weights = i_nominal_ss' / sum(i_nominal_ss);
    
    % Employment-weighted average of sectoral labor volatility
    sigma_L_avg_empweighted = sum(emp_weights .* sigma_L_sectoral);
    
    % Investment-weighted average of sectoral investment volatility (using nominal weights)
    sigma_I_avg_invweighted = sum(inv_weights .* sigma_I_sectoral);
    
    %% ===== DOMAR WEIGHT VOLATILITY =====
    % Domar weight: Domar_i = GO_i / VA_agg
    % In log deviations from SS: log(Domar_i(t)/Domar_i^ss) = q_simul(i,t) - yagg_simul(t)
    % This directly gives the cyclical component (no HP filtering needed for model)
    
    domar_simul = q_simul - repmat(yagg_simul, n_sectors, 1);  % Log Domar deviation from SS
    
    % Sectoral Domar weight volatility
    sigma_Domar_sectoral = std(domar_simul, 0, 2)';
    
    % Average Domar weight volatility (GO-weighted)
    sigma_Domar_avg = sum(go_weights .* sigma_Domar_sectoral);
    
    %% Store results
    ModelStats = struct();
    
    % Aggregate volatilities
    ModelStats.sigma_VA_agg = sigma_VA_agg;
    ModelStats.sigma_L_agg = sigma_L_agg;      % From CES aggregator (preferences)
    ModelStats.sigma_L_hc_agg = sigma_L_hc_agg; % Headcount (simple sum, comparable to data)
    ModelStats.sigma_I_agg = sigma_I_agg;
    ModelStats.sigma_M_agg = sigma_M_agg;
    
    % Other aggregate moments
    ModelStats.rho_VA_agg = rho_VA_agg;
    ModelStats.avg_pairwise_corr_VA = avg_pairwise_corr_VA;
    
    % Sectoral moments (VA-weighted averages)
    ModelStats.sigma_L_avg = sigma_L_avg;
    ModelStats.sigma_I_avg = sigma_I_avg;
    
    % Sectoral moments (own-variable weighted averages, comparable to data)
    ModelStats.sigma_L_avg_empweighted = sigma_L_avg_empweighted;
    ModelStats.sigma_I_avg_invweighted = sigma_I_avg_invweighted;
    
    % Domar weight volatility (GO-weighted average)
    ModelStats.sigma_Domar_avg = sigma_Domar_avg;
    
    % Full distributions (for diagnostics)
    ModelStats.sigma_L_sectoral = sigma_L_sectoral;
    ModelStats.sigma_I_sectoral = sigma_I_sectoral;
    ModelStats.sigma_Domar_sectoral = sigma_Domar_sectoral;
    ModelStats.corr_matrix_VA = corr_matrix_VA;
    ModelStats.va_weights = va_weights;
    ModelStats.go_weights = go_weights;
    ModelStats.emp_weights = emp_weights;
    ModelStats.inv_weights = inv_weights;
end

