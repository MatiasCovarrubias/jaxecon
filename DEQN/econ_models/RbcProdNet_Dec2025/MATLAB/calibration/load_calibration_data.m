function [calib_data, params] = load_calibration_data(params, sector_indices)
% LOAD_CALIBRATION_DATA Load and preprocess calibration data for the production network model
%
% INPUTS:
%   params         - Structure with basic model parameters (beta, eps_l, eps_c, theta, etc.)
%   sector_indices - Vector of sector indices to analyze
%
% OUTPUTS:
%   calib_data - Structure containing:
%                - conssh_data: Consumption expenditure shares
%                - capsh_data: Capital expenditure shares
%                - vash_data: Value added shares
%                - ionet_data: Input-output network matrix
%                - invnet_data: Investment network matrix
%                - labels: Labels structure with sector and client info
%   params     - Updated params structure with calibration data added

%% Input validation
validate_sector_indices(sector_indices, 37, 'load_calibration_data');

%% Load raw data
load('calibration_data.mat');
load('TFP_process.mat');

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
params.rho = modrho;
params.Sigma_A = modvcv;
params.n_sectors = n_sectors;

%% Add expenditure share data to params
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

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

