function TheoStats = compute_theoretical_statistics(oo_, M_, policies_ss, n_sectors)
% COMPUTE_THEORETICAL_STATISTICS Extract theoretical moments from Dynare solution
%
% Extracts theoretical variances and autocorrelations from Dynare's oo_.var
% computed by stoch_simul with periods=0. These are model-implied moments
% from the state-space representation, not simulation-based.
%
% INPUTS:
%   oo_         - Dynare output structure (contains oo_.var, oo_.autocorr)
%   M_          - Dynare model structure (contains M_.endo_names)
%   policies_ss - Steady state policies (for computing VA weights)
%   n_sectors   - Number of sectors
%
% OUTPUTS:
%   TheoStats - Structure with theoretical moments:
%     - sigma_C_agg: Theoretical std dev of aggregate consumption
%     - sigma_L_agg: Theoretical std dev of aggregate labor
%     - sigma_VA_agg: Theoretical std dev of aggregate GDP (value added)
%     - sigma_I_agg: Theoretical std dev of aggregate investment
%     - sigma_M_agg: Theoretical std dev of aggregate intermediates
%     - rho_VA_agg: Theoretical autocorrelation of aggregate GDP (if available)
%     - var_cov_agg: Full 5x5 variance-covariance matrix of aggregates

    TheoStats = struct();
    
    % Get variable indices
    idx = get_variable_indices(n_sectors);
    
    % The stoch_simul command lists: cagg lagg yagg iagg magg
    % So oo_.var is a 5x5 matrix in that order
    % Indices in oo_.var: 1=cagg, 2=lagg, 3=yagg, 4=iagg, 5=magg
    
    if ~isfield(oo_, 'var') || isempty(oo_.var)
        warning('compute_theoretical_statistics:NoVar', ...
            'oo_.var is empty. Run stoch_simul with periods=0 to compute theoretical moments.');
        return;
    end
    
    var_cov = oo_.var;
    
    if size(var_cov, 1) ~= 5
        % If oo_.var has different size, need to find the right indices
        % This handles the case where stoch_simul includes more variables
        warning('compute_theoretical_statistics:UnexpectedSize', ...
            'Expected 5x5 variance matrix for [cagg lagg yagg iagg magg], got %dx%d', ...
            size(var_cov, 1), size(var_cov, 2));
    end
    
    % Extract theoretical standard deviations from diagonal
    % Order in stoch_simul: cagg lagg yagg iagg magg
    sigma_C_agg = sqrt(var_cov(1, 1));
    sigma_L_agg = sqrt(var_cov(2, 2));
    sigma_VA_agg = sqrt(var_cov(3, 3));
    sigma_I_agg = sqrt(var_cov(4, 4));
    sigma_M_agg = sqrt(var_cov(5, 5));
    
    % Store in output structure
    TheoStats.sigma_C_agg = sigma_C_agg;
    TheoStats.sigma_L_agg = sigma_L_agg;
    TheoStats.sigma_VA_agg = sigma_VA_agg;
    TheoStats.sigma_I_agg = sigma_I_agg;
    TheoStats.sigma_M_agg = sigma_M_agg;
    
    % Full variance-covariance matrix of aggregates
    TheoStats.var_cov_agg = var_cov;
    
    % Correlations (off-diagonal elements normalized)
    if size(var_cov, 1) == 5
        std_vec = sqrt(diag(var_cov));
        corr_matrix = var_cov ./ (std_vec * std_vec');
        TheoStats.corr_matrix_agg = corr_matrix;
        
        % Specific correlations of interest
        TheoStats.corr_C_Y = corr_matrix(1, 3);   % Consumption-GDP correlation
        TheoStats.corr_L_Y = corr_matrix(2, 3);   % Labor-GDP correlation
        TheoStats.corr_I_Y = corr_matrix(4, 3);   % Investment-GDP correlation
    end
    
    % Autocorrelations (if computed by Dynare)
    if isfield(oo_, 'autocorr') && ~isempty(oo_.autocorr) && numel(oo_.autocorr) >= 1
        autocorr_lag1 = oo_.autocorr{1};
        
        % Diagonal elements are autocorrelations at lag 1
        if size(autocorr_lag1, 1) >= 5
            TheoStats.rho_C_agg = autocorr_lag1(1, 1);
            TheoStats.rho_L_agg = autocorr_lag1(2, 2);
            TheoStats.rho_VA_agg = autocorr_lag1(3, 3);
            TheoStats.rho_I_agg = autocorr_lag1(4, 4);
            TheoStats.rho_M_agg = autocorr_lag1(5, 5);
            TheoStats.autocorr_matrix_lag1 = autocorr_lag1;
        end
    end
    
    % Mean (for order >= 2, captures risk-adjusted mean)
    if isfield(oo_, 'mean') && ~isempty(oo_.mean)
        TheoStats.mean_aggregates = oo_.mean(1:5);
    end
    
    % Store VA weights from steady state (for reference/comparison with sectoral)
    y_ss_idx = (idx.y(1):idx.y(2)) - idx.ss_offset;
    y_ss_log = policies_ss(y_ss_idx);
    y_ss = exp(y_ss_log);
    va_weights = y_ss' / sum(y_ss);
    TheoStats.va_weights = va_weights;
    
end

