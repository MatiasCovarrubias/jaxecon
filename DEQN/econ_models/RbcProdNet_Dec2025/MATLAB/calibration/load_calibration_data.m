function [calib_data, params] = load_calibration_data(params, sector_indices, model_type)
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
%   params     - Updated params structure with calibration data added

%% Input validation
if nargin < 3
    model_type = 'VA';  % Default to value added model
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

%% Add expenditure share data to params
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

%% Compute empirical business cycle targets (HP-filtered, Törnqvist aggregation)
% Uses Törnqvist index for VA and Investment (proper aggregation)
% Uses simple sum for Employment (real units)
empirical_targets = compute_empirical_targets(VA_raw, EMP_raw, InvRaw, VAn, Invn);

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

end

function empirical_targets = compute_empirical_targets(VA_raw, EMP_raw, InvRaw, VAn, Invn)
% COMPUTE_EMPIRICAL_TARGETS Compute business cycle moments from data
%
% Uses Törnqvist index aggregation for VA and Investment (proper economic aggregation)
% Uses simple summation for Employment (already in real units)
%
% Computes HP-filtered (lambda=100) volatilities of:
%   - Aggregate GDP (value added) - Törnqvist
%   - Aggregate labor - simple sum (headcount)
%   - Aggregate investment - Törnqvist
%   - Average sectoral labor volatility (VA-weighted)
%   - Average sectoral investment volatility (VA-weighted)
%
% INPUTS:
%   VA_raw  - Real value added by sector (T x n_sectors)
%   EMP_raw - Employment by sector (T x n_sectors)
%   InvRaw  - Real investment by sector (T x n_sectors)
%   VAn     - Nominal value added by sector (T x n_sectors) - for shares
%   Invn    - Nominal investment by sector (T x n_sectors) - for shares

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
    
    %% ===== SECTORAL VOLATILITIES (average of sectoral volatilities) =====
    
    % Sectoral labor volatility
    sigma_L_sectoral = zeros(1, n_sectors);
    for i = 1:n_sectors
        log_L = log(EMP_clean(:, i));
        [~, L_cycle] = hpfilter(log_L, hp_lambda);
        sigma_L_sectoral(i) = std(L_cycle);
    end
    sigma_L_avg = sum(va_weights .* sigma_L_sectoral);
    
    % Sectoral investment volatility
    sigma_I_sectoral = zeros(1, n_sectors);
    for i = 1:n_sectors
        log_I = log(Inv_clean(:, i));
        [~, I_cycle] = hpfilter(log_I, hp_lambda);
        sigma_I_sectoral(i) = std(I_cycle);
    end
    sigma_I_avg = sum(va_weights .* sigma_I_sectoral);
    
    %% Store results
    empirical_targets = struct();
    empirical_targets.hp_lambda = hp_lambda;
    empirical_targets.va_weights = va_weights;
    empirical_targets.aggregation_method = 'tornqvist';
    
    % Aggregate volatilities (Törnqvist for VA/I, simple sum for L)
    empirical_targets.sigma_VA_agg = sigma_VA_agg;
    empirical_targets.sigma_L_agg = sigma_L_agg;
    empirical_targets.sigma_I_agg = sigma_I_agg;
    
    % Sectoral volatilities (VA-weighted average of sectoral volatilities)
    empirical_targets.sigma_L_avg = sigma_L_avg;
    empirical_targets.sigma_I_avg = sigma_I_avg;
    
    % Full sectoral distributions (for diagnostics)
    empirical_targets.sigma_L_sectoral = sigma_L_sectoral;
    empirical_targets.sigma_I_sectoral = sigma_I_sectoral;
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

