function TheoStats = compute_theoretical_statistics(oo_, M_, policies_ss, n_sectors)
% COMPUTE_THEORETICAL_STATISTICS Extract theoretical moments from Dynare solution
%
% Computes theoretical variances for EXPENDITURE-BASED aggregates using
% the state-space representation. This replaces the old approach that used
% Dynare's legacy aggregates (cagg, yagg, iagg, magg).
%
% NEW AGGREGATES (expenditure-based, matching model specification):
%   C_constP = Σ_j P̄_j × C_j        (consumption expenditure)
%   I_constP = Σ_j P̄_j × I_j^out    (investment expenditure)  
%   GDP_constP = Σ_j P̄_j × (Q_j - M_j^out) = C_constP + I_constP
%
% The variance of log-deviations is computed using first-order approximation:
%   var(log X_constP) ≈ ω' × Var(x̃) × ω
% where ω is the vector of steady-state expenditure shares.
%
% INPUTS:
%   oo_         - Dynare output structure (contains oo_.dr for state-space)
%   M_          - Dynare model structure
%   policies_ss - Steady state policies
%   n_sectors   - Number of sectors
%
% OUTPUTS:
%   TheoStats - Structure with theoretical moments for expenditure-based aggregates

    TheoStats = struct();
    
    % Get variable indices
    idx = get_variable_indices(n_sectors);
    n = n_sectors;
    
    %% ===== COMPUTE THEORETICAL VARIANCE FROM STATE-SPACE =====
    % The state-space is: x_{t+1} = A x_t + B ε_t, y_t = C x_t + D ε_t
    % Unconditional variance: Σ_x = A Σ_x A' + B Σ_ε B' (solve Lyapunov)
    %                         Σ_y = C Σ_x C' + D Σ_ε D'
    
    if ~isfield(oo_, 'dr') || ~isfield(oo_.dr, 'ghx')
        warning('compute_theoretical_statistics:NoDr', ...
            'oo_.dr not available. Cannot compute theoretical moments.');
        return;
    end
    
    % Get shock covariance matrix
    Sigma_eps = M_.Sigma_e;  % n_shocks x n_shocks
    
    % Get state transition matrices from Dynare
    % ghx: policy function w.r.t. states (n_vars x n_states)
    % ghu: policy function w.r.t. shocks (n_vars x n_shocks)
    ghx = oo_.dr.ghx;
    ghu = oo_.dr.ghu;
    
    % Get indices for states (k, a) and convert to DR order
    n_states = 2 * n;  % k and a
    
    % State indices in decision rule order
    k_dr = oo_.dr.inv_order_var(1:n);
    a_dr = oo_.dr.inv_order_var(n+1:2*n);
    state_dr_idx = [k_dr; a_dr];
    
    % Extract state transition for states only
    A_full = ghx(state_dr_idx, :);
    B_full = ghu(state_dr_idx, :);
    
    % Solve Lyapunov equation for state variance: Σ_x = A Σ_x A' + B Σ_ε B'
    BQB = B_full * Sigma_eps * B_full';
    try
        Sigma_x = dlyap(A_full, BQB);
    catch
        % Fallback: iterative solution
        Sigma_x = BQB;
        for iter = 1:500
            Sigma_x_new = A_full * Sigma_x * A_full' + BQB;
            if max(abs(Sigma_x_new(:) - Sigma_x(:))) < 1e-10
                break;
            end
            Sigma_x = Sigma_x_new;
        end
    end
    
    % Extract indices for sectoral variables in DR order
    c_dr = oo_.dr.inv_order_var(idx.c(1):idx.c(2));
    iout_dr = oo_.dr.inv_order_var(idx.iout(1):idx.iout(2));
    q_dr = oo_.dr.inv_order_var(idx.q(1):idx.q(2));
    mout_dr = oo_.dr.inv_order_var(idx.mout(1):idx.mout(2));
    l_dr = oo_.dr.inv_order_var(idx.l(1):idx.l(2));
    
    % Get policy matrices for these variables
    C_c = ghx(c_dr, :);      D_c = ghu(c_dr, :);
    C_iout = ghx(iout_dr, :); D_iout = ghu(iout_dr, :);
    C_q = ghx(q_dr, :);       D_q = ghu(q_dr, :);
    C_mout = ghx(mout_dr, :); D_mout = ghu(mout_dr, :);
    C_l = ghx(l_dr, :);       D_l = ghu(l_dr, :);
    
    % Compute variance of sectoral variables
    % Var(y) = C Σ_x C' + D Σ_ε D'
    Var_c = C_c * Sigma_x * C_c' + D_c * Sigma_eps * D_c';
    Var_iout = C_iout * Sigma_x * C_iout' + D_iout * Sigma_eps * D_iout';
    Var_q = C_q * Sigma_x * C_q' + D_q * Sigma_eps * D_q';
    Var_mout = C_mout * Sigma_x * C_mout' + D_mout * Sigma_eps * D_mout';
    Var_l = C_l * Sigma_x * C_l' + D_l * Sigma_eps * D_l';
    
    % Covariance between q and mout (needed for GDP = Q - M^out)
    Cov_q_mout = C_q * Sigma_x * C_mout' + D_q * Sigma_eps * D_mout';
    
    %% ===== COMPUTE EXPENDITURE-BASED AGGREGATE VARIANCES =====
    % For log-linearized variables, var(Σ ω_j x̃_j) = ω' Var(x̃) ω
    % where ω_j = (P̄_j × X_j^ss) / (Σ_k P̄_k × X_k^ss) is expenditure share
    
    % Get steady-state levels and prices
    p_ss_idx = (idx.p(1):idx.p(2)) - idx.ss_offset;
    c_ss_idx = (idx.c(1):idx.c(2)) - idx.ss_offset;
    iout_ss_idx = (idx.iout(1):idx.iout(2)) - idx.ss_offset;
    q_ss_idx = (idx.q(1):idx.q(2)) - idx.ss_offset;
    mout_ss_idx = (idx.mout(1):idx.mout(2)) - idx.ss_offset;
    l_ss_idx = (idx.l(1):idx.l(2)) - idx.ss_offset;
    
    p_ss = exp(policies_ss(p_ss_idx));  p_ss = p_ss(:);
    c_ss = exp(policies_ss(c_ss_idx));  c_ss = c_ss(:);
    iout_ss = exp(policies_ss(iout_ss_idx));  iout_ss = iout_ss(:);
    q_ss = exp(policies_ss(q_ss_idx));  q_ss = q_ss(:);
    mout_ss = exp(policies_ss(mout_ss_idx));  mout_ss = mout_ss(:);
    l_ss = exp(policies_ss(l_ss_idx));  l_ss = l_ss(:);
    
    % Expenditure weights for consumption
    % log(C_constP) ≈ Σ_j ω_j^C × c̃_j where ω_j^C = (P̄_j × C_j^ss) / C_constP_ss
    C_constP_ss = sum(p_ss .* c_ss);
    omega_C = (p_ss .* c_ss) / C_constP_ss;
    
    % Expenditure weights for investment
    I_constP_ss = sum(p_ss .* iout_ss);
    omega_I = (p_ss .* iout_ss) / I_constP_ss;
    
    % Expenditure weights for GDP = Q - M^out
    GDP_constP_ss = sum(p_ss .* (q_ss - mout_ss));
    omega_Q = (p_ss .* q_ss) / GDP_constP_ss;
    omega_Mout = (p_ss .* mout_ss) / GDP_constP_ss;
    
    % Labor weights (simple sum for headcount)
    L_hc_ss = sum(l_ss);
    omega_L = l_ss / L_hc_ss;
    
    % Theoretical variances of expenditure-based aggregates
    % var(log C_constP) = ω_C' × Var(c̃) × ω_C
    sigma_C_agg = sqrt(omega_C' * Var_c * omega_C);
    sigma_I_agg = sqrt(omega_I' * Var_iout * omega_I);
    
    % var(log GDP) = ω_Q' Var(q̃) ω_Q + ω_M' Var(m̃) ω_M - 2 ω_Q' Cov(q̃,m̃) ω_M
    % Note: GDP = Q - M^out, so covariance term is subtracted
    var_GDP = omega_Q' * Var_q * omega_Q + omega_Mout' * Var_mout * omega_Mout ...
              - 2 * omega_Q' * Cov_q_mout * omega_Mout;
    sigma_VA_agg = sqrt(max(var_GDP, 0));  % Ensure non-negative
    
    % Labor aggregate (headcount)
    sigma_L_agg = sqrt(omega_L' * Var_l * omega_L);
    
    % Store in output structure
    TheoStats.sigma_C_agg = sigma_C_agg;
    TheoStats.sigma_L_agg = sigma_L_agg;
    TheoStats.sigma_VA_agg = sigma_VA_agg;
    TheoStats.sigma_I_agg = sigma_I_agg;
    
    % Legacy aggregates from Dynare (for comparison/diagnostics)
    if isfield(oo_, 'var') && ~isempty(oo_.var) && size(oo_.var, 1) >= 5
        var_cov = oo_.var;
        TheoStats.sigma_C_legacy = sqrt(var_cov(1, 1));  % cagg (utility CES)
        TheoStats.sigma_VA_legacy = sqrt(var_cov(3, 3)); % yagg (primary factors)
        TheoStats.sigma_I_legacy = sqrt(var_cov(4, 4));  % iagg (with P_k prices)
        TheoStats.sigma_M_agg = sqrt(var_cov(5, 5));     % magg
        
        % Full variance-covariance matrix of legacy aggregates
        TheoStats.var_cov_agg_legacy = var_cov;
        
        % Legacy correlations
        std_vec = sqrt(diag(var_cov));
        corr_matrix = var_cov ./ (std_vec * std_vec');
        TheoStats.corr_matrix_agg_legacy = corr_matrix;
    end
    
    % Compute expenditure-based aggregate covariances for correlation
    % Cov(C, I) for the new aggregates
    Cov_c_iout = C_c * Sigma_x * C_iout' + D_c * Sigma_eps * D_iout';
    cov_CI = omega_C' * Cov_c_iout * omega_I;
    TheoStats.corr_C_I = cov_CI / (sigma_C_agg * sigma_I_agg);
    
    % Autocorrelations (legacy, from Dynare)
    if isfield(oo_, 'autocorr') && ~isempty(oo_.autocorr) && numel(oo_.autocorr) >= 1
        autocorr_lag1 = oo_.autocorr{1};
        
        % These are autocorrelations of legacy aggregates
        if size(autocorr_lag1, 1) >= 5
            TheoStats.rho_C_agg_legacy = autocorr_lag1(1, 1);
            TheoStats.rho_L_agg_legacy = autocorr_lag1(2, 2);
            TheoStats.rho_VA_agg_legacy = autocorr_lag1(3, 3);
            TheoStats.rho_I_agg_legacy = autocorr_lag1(4, 4);
            TheoStats.rho_M_agg_legacy = autocorr_lag1(5, 5);
        end
    end
    
    % Store expenditure weights
    TheoStats.omega_C = omega_C;
    TheoStats.omega_I = omega_I;
    TheoStats.omega_Q = omega_Q;
    TheoStats.omega_Mout = omega_Mout;
    
    % Store VA weights from steady state
    y_ss_idx = (idx.y(1):idx.y(2)) - idx.ss_offset;
    y_ss_log = policies_ss(y_ss_idx);
    y_ss = exp(y_ss_log);  y_ss = y_ss(:);
    va_weights = y_ss' / sum(y_ss);
    TheoStats.va_weights = va_weights;
    
end

