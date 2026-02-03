function [ModData, params] = calibrate_steady_state(params, opts)
% CALIBRATE_STEADY_STATE Calibrates the steady state of the production network RBC model
%
% This function runs a multi-stage calibration procedure:
%   1. Solve Steady State under Cobb-Douglas (high sigmas)
%   2. Lower sigma_l to target value
%   3. Match value added and IO network expenditure shares
%   4. Match all expenditure shares (consumption, capital, investment network)
%   5. Homotopy on eps_l (Frisch elasticity) from 0.5 to target value
%
% INPUTS:
%   params  - Structure with model parameters and calibration targets
%             Required fields:
%               - beta, eps_c, theta, delta, rho, Sigma_A
%               - n_sectors
%               - conssh_data, capsh_data, vash_data, ionet_data, invnet_data
%             Optional fields (defaults provided):
%               - sigma_c, sigma_m, sigma_q, sigma_y, sigma_I, sigma_l (target values)
%               - eps_l (target Frisch elasticity, default 0.5 = no homotopy)
%
%   opts    - Structure with calibration options (optional)
%             Fields:
%               - gridpoints: Number of homotopy steps (default: 8)
%               - verbose: Print progress (default: true)
%               - sol_guess_file: File with initial guess (default: 'SS_CDsolution_norm_permanent.mat')
%               - fsolve_options: Options for fsolve (default: TolX=1e-10, etc.)
%
% OUTPUTS:
%   ModData - Structure with steady state solution containing:
%               - parameters: calibrated parameters for Dynare
%               - policies_ss: steady state policy variables
%               - endostates_ss: steady state endogenous states (log capital)
%               - Cagg_ss, Lagg_ss, Yagg_ss, Iagg_ss, Magg_ss, V_ss
%   params  - Updated params structure with calibrated values

%% Input validation
validate_params(params, {'n_sectors', 'beta', 'delta', 'rho', 'Sigma_A', ...
    'conssh_data', 'capsh_data', 'vash_data', 'ionet_data', 'invnet_data'}, ...
    'calibrate_steady_state');

%% Set defaults for options
if nargin < 2
    opts = struct();
end

opts = set_default(opts, 'gridpoints', 8);
opts = set_default(opts, 'verbose', true);
opts = set_default(opts, 'sol_guess_file', 'SS_CDsolution_norm_permanent.mat');
opts = set_default(opts, 'fsolve_options', optimset('Display','iter','TolX',1e-10,'TolFun',1e-10,...
    'MaxFunEvals',10000000,'MaxIter',10000));

%% Extract target elasticities (use defaults if not specified)
params = set_default(params, 'sigma_c', 0.5);
params = set_default(params, 'sigma_m', 0.01);
params = set_default(params, 'sigma_q', 0.5);
params = set_default(params, 'sigma_y', 0.8);
params = set_default(params, 'sigma_I', 0.5);
params = set_default(params, 'sigma_l', 0.1);
params = set_default(params, 'eps_l', 0.5);

sigma_c_target = params.sigma_c;
sigma_m_target = params.sigma_m;
sigma_q_target = params.sigma_q;
sigma_y_target = params.sigma_y;
sigma_I_target = params.sigma_I;
sigma_l_target = params.sigma_l;
eps_l_target = params.eps_l;

n_sectors = params.n_sectors;
gridpoints = opts.gridpoints;
fsolve_opts = opts.fsolve_options;

%% ========== STAGE 1: Solve Steady State under Cobb-Douglas ==========
if opts.verbose
    fprintf('\n');
    fprintf('  ┌─ STAGE 1: Cobb-Douglas Initialization ───────────────────────┐\n');
    fprintf('  │  σ = 0.99 for all elasticities (CD approximation)            │\n');
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
end

% Set elasticities close to 1 (Cobb-Douglas)
params.sigma_c = 0.5;
params.sigma_m = 0.99;
params.sigma_q = 0.99;
params.sigma_y = 0.99;
params.sigma_I = 0.99;
params.sigma_l = 0.99;
params.eps_l = 0.5;  % Start with eps_l = 0.5, homotopy to target in Stage 5

% Intensity shares equal to expenditure shares in CD case
params.xi = params.conssh_data;
params.alpha = params.capsh_data;
params.mu = params.vash_data;
params.Gamma_M = params.ionet_data;
params.Gamma_I = params.invnet_data;

% Load initial guess (11*n + 2 variables: policies + Cagg + Lagg)
if exist(opts.sol_guess_file, 'file')
    loaded = load(opts.sol_guess_file);
    if isfield(loaded, 'sol_init')
        sol_guess = loaded.sol_init;
    else
        sol_guess = zeros([11*n_sectors+2, 1]);
    end
else
    sol_guess = zeros([11*n_sectors+2, 1]);
end

% Solve
tic_stage1 = tic;
fh_compStSt = @(x) ProdNetRbc_SS(x, params, 0);
[sol_init, ~, exfl] = fsolve(fh_compStSt, sol_guess, fsolve_opts);
[~, ModData] = ProdNetRbc_SS(sol_init, params, 0);

if opts.verbose
    fprintf('\n  Stage 1 completed: %s (%.2f s)\n', exit_flag_str(exfl), toc(tic_stage1));
    fprintf('    θ = %.4f\n', ModData.parameters.partheta);
end

%% ========== STAGE 2: Lower sigma_l ==========
if opts.verbose
    fprintf('\n');
    fprintf('  ┌─ STAGE 2: Lowering σ_l ──────────────────────────────────────┐\n');
    fprintf('  │  Homotopy: 0.90 → %.2f (%d steps)                            │\n', sigma_l_target, gridpoints);
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
end

sol_guess = sol_init;
sigma_l_grid = linspace(0.9, sigma_l_target, gridpoints);

tic_stage2 = tic;
for i = 1:gridpoints
    tic_iter = tic;
    params.sigma_l = sigma_l_grid(i);
    fh_compStSt = @(x) ProdNetRbc_SS(x, params, 0);
    [sol_init, ~, exfl] = fsolve(fh_compStSt, sol_guess, fsolve_opts);
    [~, ModData] = ProdNetRbc_SS(sol_init, params, 0);
    sol_guess = sol_init;
    
    if opts.verbose
        fprintf('    [%d/%d] σ_l = %.3f  →  %s (%.2fs)\n', i, gridpoints, params.sigma_l, exit_flag_str(exfl), toc(tic_iter));
    end
end

if opts.verbose
    fprintf('\n  Stage 2 completed (%.2f s) | θ = %.4f\n', toc(tic_stage2), ModData.parameters.partheta);
end

%% ========== STAGE 3: Match value added and IO network shares ==========
if opts.verbose
    fprintf('\n');
    fprintf('  ┌─ STAGE 3: Value Added & IO Network Shares ───────────────────┐\n');
    fprintf('  │  Matching μ (VA) and Γ_M (IO network)                        │\n');
    fprintf('  │  Homotopy: σ_m, σ_q: 0.99 → %.2f, %.2f (%d steps)           │\n', sigma_m_target, sigma_q_target, gridpoints);
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
end

% Build solution guess for extended problem
mu_guess = params.vash_data;
Gamma_M_guess = params.ionet_data(1:n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
sol_guess = [sol_init; log(mu_guess); log(Gamma_M_guess)];

% Grids for homotopy
sigma_m_grid = linspace(0.99, sigma_m_target, gridpoints);
sigma_q_grid = linspace(0.99, sigma_q_target, gridpoints);

tic_stage3 = tic;
for i = 1:gridpoints
    tic_iter = tic;
    params.sigma_m = sigma_m_grid(i);
    params.sigma_q = sigma_q_grid(i);
    fh_compStSt = @(x) ProdNetRbc_SS_vaioshares(x, params, 0);
    [sol_partial, ~, exfl] = fsolve(fh_compStSt, sol_guess, fsolve_opts);
    [~, ModData] = ProdNetRbc_SS_vaioshares(sol_partial, params, 0);
    sol_guess = sol_partial;
    
    if opts.verbose
        fprintf('    [%d/%d] σ_m = %.3f, σ_q = %.3f  →  %s (%.2fs)\n', ...
            i, gridpoints, params.sigma_m, params.sigma_q, exit_flag_str(exfl), toc(tic_iter));
    end
end

if opts.verbose
    fprintf('\n  Stage 3 completed (%.2f s) | θ = %.4f\n', toc(tic_stage3), ModData.parameters.partheta);
end

%% ========== STAGE 4: Match all expenditure shares ==========
if opts.verbose
    fprintf('\n');
    fprintf('  ┌─ STAGE 4: All Expenditure Shares ────────────────────────────┐\n');
    fprintf('  │  Matching ξ (cons), α (cap), Γ_I (inv network)               │\n');
    fprintf('  │  Homotopy: σ_c, σ_y, σ_I → %.2f, %.2f, %.2f (%d steps)      │\n', sigma_c_target, sigma_y_target, sigma_I_target, gridpoints);
    fprintf('  └────────────────────────────────────────────────────────────────┘\n');
end

% Build solution guess for fully extended problem
xi_guess = params.conssh_data;
mu_guess = ModData.parameters.parmu;
alpha_guess = params.capsh_data;
Gamma_M_guess = ModData.parameters.parGamma_M(1:n_sectors-1, :);
Gamma_M_guess = Gamma_M_guess(:);
Gamma_I_guess = params.invnet_data(1:n_sectors-1, :);
Gamma_I_guess = Gamma_I_guess(:);
sol_guess = [sol_partial(1:11*n_sectors+2); log(xi_guess); log(mu_guess); ...
             log(alpha_guess); log(Gamma_M_guess); log(Gamma_I_guess)];

% Grids for homotopy
sigma_c_grid = linspace(0.5, sigma_c_target, gridpoints);
sigma_y_grid = linspace(0.99, sigma_y_target, gridpoints);
sigma_I_grid = linspace(0.99, sigma_I_target, gridpoints);

tic_stage4 = tic;
for i = 1:gridpoints
    tic_iter = tic;
    params.sigma_c = sigma_c_grid(i);
    params.sigma_y = sigma_y_grid(i);
    params.sigma_I = sigma_I_grid(i);
    fh_compStSt = @(x) ProdNetRbc_SS_expshares(x, params, 0);
    [sol_final, ~, exfl] = fsolve(fh_compStSt, sol_guess, fsolve_opts);
    [~, ModData] = ProdNetRbc_SS_expshares(sol_final, params, 0);
    sol_guess = sol_final;
    
    if opts.verbose
        fprintf('    [%d/%d] σ_c = %.3f, σ_y = %.3f, σ_I = %.3f  →  %s (%.2fs)\n', ...
            i, gridpoints, params.sigma_c, params.sigma_y, params.sigma_I, exit_flag_str(exfl), toc(tic_iter));
    end
end

if opts.verbose
    fprintf('\n  Stage 4 completed (%.2f s) | θ = %.4f\n', toc(tic_stage4), ModData.parameters.partheta);
end

%% ========== STAGE 5: Homotopy on eps_l (Frisch elasticity) ==========
% Starting eps_l is 0.5 (set in Stage 1), traverse to target eps_l
eps_l_start = 0.5;

if eps_l_target ~= eps_l_start
    if opts.verbose
        fprintf('\n');
        fprintf('  ┌─ STAGE 5: Homotopy on ε_l (Frisch elasticity) ─────────────────┐\n');
        fprintf('  │  Homotopy: %.2f → %.2f (%d steps)                              │\n', eps_l_start, eps_l_target, gridpoints);
        fprintf('  └────────────────────────────────────────────────────────────────┘\n');
    end
    
    % Use solution from Stage 4 as starting point
    sol_guess = sol_final;
    eps_l_grid = linspace(eps_l_start, eps_l_target, gridpoints);
    
    tic_stage5 = tic;
    for i = 1:gridpoints
        tic_iter = tic;
        params.eps_l = eps_l_grid(i);
        fh_compStSt = @(x) ProdNetRbc_SS_expshares(x, params, 0);
        [sol_final, ~, exfl] = fsolve(fh_compStSt, sol_guess, fsolve_opts);
        [~, ModData] = ProdNetRbc_SS_expshares(sol_final, params, 0);
        sol_guess = sol_final;
        
        if opts.verbose
            fprintf('    [%d/%d] ε_l = %.3f  →  %s (%.2fs)\n', i, gridpoints, params.eps_l, exit_flag_str(exfl), toc(tic_iter));
        end
    end
    
    if opts.verbose
        fprintf('\n  Stage 5 completed (%.2f s) | θ = %.4f\n', toc(tic_stage5), ModData.parameters.partheta);
    end
else
    if opts.verbose
        fprintf('\n  Stage 5 skipped (ε_l already at target: %.2f)\n', eps_l_target);
    end
end

if opts.verbose
    fprintf('\n');
    fprintf('  ════════════════════════════════════════════════════════════════\n');
    fprintf('    ✓ STEADY STATE CALIBRATION COMPLETE\n');
    fprintf('  ════════════════════════════════════════════════════════════════\n');
end

end

%% ==================== Helper Functions ====================

function str = exit_flag_str(exfl)
    switch exfl
        case 1
            str = 'Converged';
        case 2
            str = 'Change in x < TolX';
        case 3
            str = 'Change in residual < TolFun';
        case 4
            str = 'Magnitude of search direction < TolX';
        case 0
            str = 'Max iterations exceeded';
        case -1
            str = 'Output function terminated';
        case -2
            str = 'Converging to non-solution';
        case -3
            str = 'Trust region radius too small';
        otherwise
            str = sprintf('Exit flag: %d', exfl);
    end
end

