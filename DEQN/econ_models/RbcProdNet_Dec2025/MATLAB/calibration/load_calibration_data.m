function [calib_data, params] = load_calibration_data(params, sector_indices, model_type, shock_scaling)
% LOAD_CALIBRATION_DATA Load and preprocess calibration data for the production network model
%
% INPUTS:
%   params         - Structure with basic model parameters (beta, eps_l, eps_c, theta, etc.)
%   sector_indices - Vector of sector indices to analyze
%   model_type     - Model type: 'VA' (value added), 'GO' (gross output), or 'GO_noVA'
%                    Determines which TFP process data to load:
%                    - 'VA': modrho, modvcv (no suffix)
%                    - 'GO': modrho_GO, modvcv_GO
%                    - 'GO_noVA': modrho_GO_noVA, modvcv_GO_noVA
%   shock_scaling  - (Optional) Structure for scaling shock volatility:
%                    - sectors: Vector of sector indices to scale (e.g., [1] for Mining)
%                    - factor: Multiplicative factor for shock std dev
%                    If factor=2, shock std dev doubles (variance × 4, covariances × 2)
%
% OUTPUTS:
%   calib_data - Structure containing:
%                - conssh_data: Consumption expenditure shares
%                - capsh_data: Capital expenditure shares
%                - vash_data: Value added shares
%                - ionet_data: Input-output network matrix
%                - invnet_data: Investment network matrix
%                - labels: Labels structure with sector and client info
%                - empirical_targets: Business cycle moments from data
%                - model_type: The model type used
%                - shock_scaling: Applied shock scaling (if any)
%   params     - Updated params structure with calibration data added

%% Input validation
if nargin < 3
    model_type = 'VA';  % Default to value added model
end
if nargin < 4
    shock_scaling = struct('sectors', [], 'factor', 1.0);
end
validate_sector_indices(sector_indices, 37, 'load_calibration_data');

valid_model_types = {'VA', 'GO', 'GO_noVA'};
if ~ismember(model_type, valid_model_types)
    error('load_calibration_data:InvalidModelType', ...
        'model_type must be one of: %s', strjoin(valid_model_types, ', '));
end

%% Load raw data
load('calibration_data.mat');
load('TFP_process.mat');

%% Load BEA sectoral data for empirical moments (same as calibration_moments.m)
% This file contains both real (VA_raw, InvRaw) and nominal (VAn, Invn) sectoral data
% Nominal data is needed for Törnqvist share computation
if exist('beadat_37sec.mat', 'file')
    load('beadat_37sec.mat');
else
    error('load_calibration_data:MissingBEAData', ...
        ['beadat_37sec.mat not found.\n' ...
         'This file is required for Törnqvist aggregation (contains VAn, Invn for shares).\n' ...
         'Place it in the root MATLAB folder.']);
end

n_sectors = 37;

%% Process consumption expenditure shares
conssh_data = mean(Cons47bea ./ repmat(sum(Cons47bea, 2), 1, n_sectors))';

%% Process capital expenditure shares
capshbea(capshbea < 0.05) = 0.05;
capsh_data = mean(capshbea, 2);

%% Process value added share of gross output
vash_data = (mean(VA47bea ./ GO47bea))';

%% Process investment matrix
invnet_data = mean(invmat, 3);
invnet_data(invnet_data < 0.001) = 0.001;
invnet_data = invnet_data ./ sum(invnet_data, 1);

%% Process input-output matrix
ionet_data = mean(IOmat47bea ./ repmat(sum(IOmat47bea), [n_sectors 1 1]), 3);
ionet_data(ionet_data < 0.001) = 0.001;
ionet_data = ionet_data ./ sum(ionet_data, 1);

%% Update params with depreciation and TFP process
params.delta = mean(depratbea, 2);
params.n_sectors = n_sectors;
params.model_type = model_type;

% Select TFP process based on model_type
switch model_type
    case 'VA'
        params.rho = modrho;
        params.Sigma_A = modvcv;
    case 'GO'
        params.rho = modrho_GO;
        params.Sigma_A = modvcv_GO;
    case 'GO_noVA'
        params.rho = modrho_GO_noVA;
        params.Sigma_A = modvcv_GO_noVA;
end

%% Apply shock scaling if specified
% Scales the VCV matrix to increase/decrease shock volatility for specific sectors
% Math: To scale shock std dev by factor f for sector i:
%   - Var(f*ε_i) = f² * Var(ε_i)  →  Σ(i,i) *= f²
%   - Cov(f*ε_i, ε_j) = f * Cov(ε_i, ε_j)  →  Σ(i,j) *= f, Σ(j,i) *= f
% This is equivalent to: Σ_new = D * Σ * D, where D = diag with D(i,i)=f for scaled sectors
if ~isempty(shock_scaling.sectors) && shock_scaling.factor ~= 1.0
    % Validate sector indices
    validate_sector_indices(shock_scaling.sectors, n_sectors, 'shock_scaling');
    
    % Build diagonal scaling matrix D
    D = eye(n_sectors);
    for i = 1:numel(shock_scaling.sectors)
        sec_idx = shock_scaling.sectors(i);
        D(sec_idx, sec_idx) = shock_scaling.factor;
    end
    
    % Apply scaling: Σ_new = D * Σ * D
    params.Sigma_A = D * params.Sigma_A * D;
    
    % Store scaling info for reference
    params.shock_scaling = shock_scaling;
end

%% Add expenditure share data to params
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

%% Load aggregate consumption from NIPA (Real GDP Components)
% This is aggregate real PCE, already chain-weighted by BEA
% Used for aggregate consumption volatility benchmark
Cons_agg = [];
if exist('Real GDP Components.xls', 'file')
    Cons_agg = xlsread('Real GDP Components.xls', 1, 'C9:BU9')';
elseif exist('Data/Real GDP Components.xls', 'file')
    Cons_agg = xlsread('Data/Real GDP Components.xls', 1, 'C9:BU9')';
end

%% Compute empirical business cycle targets (HP-filtered, Törnqvist aggregation)
% Uses Törnqvist index for VA and Investment (proper aggregation)
% Uses simple sum for Employment (real units)
% GO47bea is used for Domar weight volatility
empirical_targets = compute_empirical_targets(VA_raw, EMP_raw, InvRaw, VAn, Invn, GO47bea, VA47bea, Cons_agg);

%% Compute client indices and rankings
[client_indices, ranking] = compute_client_rankings(ionet_data, sector_indices, n_sectors);

%% Create labels structure
sector_label_struct = SectorLabel(sector_indices);
client_label_struct = SectorLabel(client_indices);

labels = struct();
labels.sector_indices = sector_indices;
labels.sector_labels = sector_label_struct.display;
labels.sector_labels_latex = sector_label_struct.latex;
labels.sector_labels_filename = sector_label_struct.filename;
labels.client_indices = client_indices;
labels.client_labels = client_label_struct.display;
labels.client_labels_latex = client_label_struct.latex;
labels.client_labels_filename = client_label_struct.filename;
labels.ranking = ranking;

%% Build output structure
calib_data = struct();
calib_data.conssh_data = conssh_data;
calib_data.capsh_data = capsh_data;
calib_data.vash_data = vash_data;
calib_data.ionet_data = ionet_data;
calib_data.invnet_data = invnet_data;
calib_data.labels = labels;
calib_data.empirical_targets = empirical_targets;
calib_data.model_type = model_type;
calib_data.shock_scaling = shock_scaling;

end

function empirical_targets = compute_empirical_targets(VA_raw, EMP_raw, InvRaw, VAn, Invn, GO_raw, VA_for_domar, Cons_agg)
% COMPUTE_EMPIRICAL_TARGETS Compute business cycle moments from data
%
% Uses Törnqvist index aggregation for VA and Investment (proper economic aggregation)
% Uses simple summation for Employment (already in real units)
%
% Computes HP-filtered (lambda=100) volatilities of:
%   - Aggregate GDP (value added) - Törnqvist
%   - Aggregate consumption - NIPA real PCE (chain-weighted by BEA)
%   - Aggregate labor - simple sum (headcount)
%   - Aggregate investment - Törnqvist
%   - Average sectoral labor volatility (VA-weighted and employment-weighted)
%   - Average sectoral investment volatility (VA-weighted and investment-weighted)
%   - Average Domar weight volatility (GO-weighted)
%
% INPUTS:
%   VA_raw       - Real value added by sector (T1 x n_sectors) - for business cycle moments
%   EMP_raw      - Employment by sector (T1 x n_sectors)
%   InvRaw       - Real investment by sector (T1 x n_sectors)
%   VAn          - Nominal value added by sector (T1 x n_sectors) - for shares
%   Invn         - Nominal investment by sector (T1 x n_sectors) - for shares
%   GO_raw       - Real gross output by sector (T2 x n_sectors) - for Domar weights
%   VA_for_domar - Real value added matching GO_raw dimensions (T2 x n_sectors) - for Domar denominator
%   Cons_agg     - Aggregate real consumption from NIPA (T x 1), can be empty

    hp_lambda = 100;  % Standard for annual data
    epsilon = 1e-10;  % Floor for log computation
    
    % Clean data: replace zeros/negatives with small positive value
    VA_clean = max(VA_raw, epsilon);
    EMP_clean = max(EMP_raw, epsilon);
    Inv_clean = max(InvRaw, epsilon);
    VAn_clean = max(VAn, epsilon);
    Invn_clean = max(Invn, epsilon);
    
    [yrnum, n_sectors] = size(VA_clean);
    
    % Compute time-varying shares from nominal data (for Törnqvist)
    VAsh = VAn_clean ./ repmat(sum(VAn_clean, 2), 1, n_sectors);
    Invsh = Invn_clean ./ repmat(sum(Invn_clean, 2), 1, n_sectors);
    
    % Compute time-average weights (for sectoral averages)
    va_weights = mean(VAsh, 1);
    va_weights = va_weights / sum(va_weights);
    
    % Employment weights (employment is in natural units - persons/FTE)
    EMPsh = EMP_clean ./ repmat(sum(EMP_clean, 2), 1, n_sectors);
    emp_weights = mean(EMPsh, 1);
    emp_weights = emp_weights / sum(emp_weights);
    
    % Nominal investment weights (proper economic shares)
    inv_weights = mean(Invsh, 1);
    inv_weights = inv_weights / sum(inv_weights);
    
    %% ===== AGGREGATE VARIABLES VIA TÖRNQVIST INDEX =====
    % Törnqvist: log growth = sum_j [ 0.5*(s_{j,t} + s_{j,t-1}) * (log y_{j,t} - log y_{j,t-1}) ]
    
    % Aggregate VA (Törnqvist)
    aggVA = ones(yrnum, 1);
    for t = 1:yrnum-1
        aggVAgr = 0;
        for j = 1:n_sectors
            growth_j = log(VA_clean(t+1, j)) - log(VA_clean(t, j));
            weight_j = 0.5 * (VAsh(t, j) + VAsh(t+1, j));
            aggVAgr = aggVAgr + growth_j * weight_j;
        end
        aggVA(t+1) = aggVA(t) * exp(aggVAgr);
    end
    
    % Aggregate Investment (Törnqvist)
    aggInv = ones(yrnum, 1);
    for t = 1:yrnum-1
        aggInvgr = 0;
        for j = 1:n_sectors
            growth_j = log(Inv_clean(t+1, j)) - log(Inv_clean(t, j));
            weight_j = 0.5 * (Invsh(t, j) + Invsh(t+1, j));
            aggInvgr = aggInvgr + growth_j * weight_j;
        end
        aggInv(t+1) = aggInv(t) * exp(aggInvgr);
    end
    
    % Aggregate Employment (simple sum - already in real units)
    aggEMP = sum(EMP_clean, 2);
    
    %% ===== HP FILTER AND COMPUTE VOLATILITIES =====
    
    % Aggregate VA volatility
    log_aggVA = log(aggVA);
    [~, VA_agg_cycle] = hpfilter(log_aggVA, hp_lambda);
    sigma_VA_agg = std(VA_agg_cycle);
    
    % Aggregate Investment volatility
    log_aggInv = log(aggInv);
    [~, I_agg_cycle] = hpfilter(log_aggInv, hp_lambda);
    sigma_I_agg = std(I_agg_cycle);
    
    % Aggregate Employment volatility
    log_aggEMP = log(aggEMP);
    [~, L_agg_cycle] = hpfilter(log_aggEMP, hp_lambda);
    sigma_L_agg = std(L_agg_cycle);
    
    % Aggregate Consumption volatility (NIPA real PCE - already aggregated by BEA)
    % Note: This is household consumption, not sectoral consumption expenditure
    if ~isempty(Cons_agg)
        Cons_clean = max(Cons_agg, epsilon);
        log_C_agg = log(Cons_clean);
        [~, C_agg_cycle] = hpfilter(log_C_agg, hp_lambda);
        sigma_C_agg = std(C_agg_cycle);
    else
        sigma_C_agg = NaN;
    end
    
    %% ===== SECTORAL VOLATILITIES (average of sectoral volatilities) =====
    
    % Sectoral labor volatility
    sigma_L_sectoral = zeros(1, n_sectors);
    for i = 1:n_sectors
        log_L = log(EMP_clean(:, i));
        [~, L_cycle] = hpfilter(log_L, hp_lambda);
        sigma_L_sectoral(i) = std(L_cycle);
    end
    sigma_L_avg = sum(va_weights .* sigma_L_sectoral);
    sigma_L_avg_empweighted = sum(emp_weights .* sigma_L_sectoral);
    
    % Sectoral investment volatility
    sigma_I_sectoral = zeros(1, n_sectors);
    for i = 1:n_sectors
        log_I = log(Inv_clean(:, i));
        [~, I_cycle] = hpfilter(log_I, hp_lambda);
        sigma_I_sectoral(i) = std(I_cycle);
    end
    sigma_I_avg = sum(va_weights .* sigma_I_sectoral);
    sigma_I_avg_invweighted = sum(inv_weights .* sigma_I_sectoral);
    
    %% ===== DOMAR WEIGHT VOLATILITY =====
    % Domar weight: Domar_i(t) = GO_i(t) / VA_agg(t)
    % Measures each sector's sales relative to aggregate GDP
    % Note: GO_raw and VA_for_domar have matching dimensions (may differ from VA_raw)
    GO_clean = max(GO_raw, epsilon);
    VA_domar_clean = max(VA_for_domar, epsilon);
    [T_domar, n_sectors_domar] = size(GO_clean);
    
    % Compute GO-based weights (time-average)
    go_weights = mean(GO_clean, 1) / sum(mean(GO_clean, 1));
    
    % Aggregate VA (simple sum for Domar denominator)
    aggVA_domar = sum(VA_domar_clean, 2);
    
    % Compute Domar weights for each sector and time period
    Domar = GO_clean ./ repmat(aggVA_domar, 1, n_sectors_domar);  % (T_domar x n_sectors)
    
    % HP-filter log Domar weights and compute volatility for each sector
    sigma_Domar_sectoral = zeros(1, n_sectors_domar);
    for i = 1:n_sectors_domar
        log_Domar = log(Domar(:, i));
        [~, Domar_cycle] = hpfilter(log_Domar, hp_lambda);
        sigma_Domar_sectoral(i) = std(Domar_cycle);
    end
    
    % Average Domar weight volatility (GO-weighted)
    sigma_Domar_avg = sum(go_weights .* sigma_Domar_sectoral);
    
    %% Store results
    empirical_targets = struct();
    empirical_targets.hp_lambda = hp_lambda;
    empirical_targets.va_weights = va_weights;
    empirical_targets.go_weights = go_weights;
    empirical_targets.emp_weights = emp_weights;
    empirical_targets.inv_weights = inv_weights;
    empirical_targets.aggregation_method = 'tornqvist';
    
    % Aggregate volatilities (Törnqvist for VA/I, simple sum for L, NIPA for C)
    empirical_targets.sigma_VA_agg = sigma_VA_agg;
    empirical_targets.sigma_C_agg = sigma_C_agg;
    empirical_targets.sigma_L_agg = sigma_L_agg;
    empirical_targets.sigma_I_agg = sigma_I_agg;
    
    % Sectoral volatilities (VA-weighted average of sectoral volatilities)
    empirical_targets.sigma_L_avg = sigma_L_avg;
    empirical_targets.sigma_I_avg = sigma_I_avg;
    
    % Sectoral volatilities with own-variable weights
    % sigma_L_avg_empweighted: average labor volatility weighted by employment shares
    % sigma_I_avg_invweighted: average investment volatility weighted by nominal investment shares
    empirical_targets.sigma_L_avg_empweighted = sigma_L_avg_empweighted;
    empirical_targets.sigma_I_avg_invweighted = sigma_I_avg_invweighted;
    
    % Domar weight volatility (GO-weighted average of sectoral Domar volatilities)
    empirical_targets.sigma_Domar_avg = sigma_Domar_avg;
    
    % Full sectoral distributions (for diagnostics)
    empirical_targets.sigma_L_sectoral = sigma_L_sectoral;
    empirical_targets.sigma_I_sectoral = sigma_I_sectoral;
    empirical_targets.sigma_Domar_sectoral = sigma_Domar_sectoral;
end

function [client_indices, ranking] = compute_client_rankings(ionet_data, sector_indices, n_sectors)
% COMPUTE_CLIENT_RANKINGS Extract maximum client and ranking for each sector
%
% For each sector, finds its largest client (excluding self) and computes
% the ranking of all sectors by their share of this sector's output.

    n_analyzed = numel(sector_indices);
    client_indices = zeros(n_analyzed, 1);
    ranking = zeros(n_analyzed, n_sectors);
    
    for i = 1:n_analyzed
        s_idx = sector_indices(i);
        
        % Exclude own sector when finding max client
        ionet_without_sector = [ionet_data(s_idx, 1:s_idx-1), ionet_data(s_idx, s_idx+1:end)];
        [~, col_index] = max(ionet_without_sector);
        if col_index >= s_idx
            col_index = col_index + 1;
        end
        client_indices(i) = col_index;
        
        % Compute ranking
        shares_vector = ionet_data(s_idx, :);
        sorted_ionet = sort(shares_vector, 'descend');
        [~, rank] = ismember(shares_vector, sorted_ionet);
        
        % Place own sector first in ranking if not already
        if rank(s_idx) ~= 1
            rank(rank < rank(s_idx)) = rank(rank < rank(s_idx)) + 1;
            rank(s_idx) = 1;
        end
        ranking(i, :) = rank;
    end
end

