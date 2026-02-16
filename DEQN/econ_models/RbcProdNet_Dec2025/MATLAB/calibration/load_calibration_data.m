function [calib_data, params] = load_calibration_data(params, sector_indices, model_type, shock_scaling)

%% Defaults
if nargin < 3, model_type = 'VA'; end
if nargin < 4, shock_scaling = struct('sectors', [], 'factor', 1.0); end

validate_sector_indices(sector_indices, 37, 'load_calibration_data');
assert(ismember(model_type, {'VA', 'GO', 'GO_noVA'}), 'Invalid model_type: %s', model_type);

%% Load raw data
load('calibration_data.mat');
load('TFP_process.mat');

if ~exist('beadat_37sec.mat', 'file')
    error('load_calibration_data:MissingBEAData', 'beadat_37sec.mat not found.');
end
load('beadat_37sec.mat');

n_sectors = 37;

%% Process shares and networks
conssh_data = mean(Cons47bea ./ repmat(sum(Cons47bea, 2), 1, n_sectors))';

capshbea(capshbea < 0.05) = 0.05;
capsh_data = mean(capshbea, 2);

vash_data = (mean(VA47bea ./ GO47bea))';

invnet_data = mean(invmat, 3);
invnet_data(invnet_data < 0.001) = 0.001;
invnet_data = invnet_data ./ sum(invnet_data, 1);

ionet_data = mean(IOmat47bea ./ repmat(sum(IOmat47bea), [n_sectors 1 1]), 3);
ionet_data(ionet_data < 0.001) = 0.001;
ionet_data = ionet_data ./ sum(ionet_data, 1);

%% TFP process
params.delta = mean(depratbea, 2);
params.n_sectors = n_sectors;
params.model_type = model_type;

switch model_type
    case 'VA',      params.rho = modrho;         params.Sigma_A = modvcv;
    case 'GO',      params.rho = modrho_GO;      params.Sigma_A = modvcv_GO;
    case 'GO_noVA', params.rho = modrho_GO_noVA; params.Sigma_A = modvcv_GO_noVA;
end

%% Shock scaling
if ~isempty(shock_scaling.sectors) && shock_scaling.factor ~= 1.0
    validate_sector_indices(shock_scaling.sectors, n_sectors, 'shock_scaling');
    D = eye(n_sectors);
    for i = 1:numel(shock_scaling.sectors)
        D(shock_scaling.sectors(i), shock_scaling.sectors(i)) = shock_scaling.factor;
    end
    params.Sigma_A = D * params.Sigma_A * D;
    params.shock_scaling = shock_scaling;
end

%% Store in params
params.conssh_data = conssh_data;
params.capsh_data = capsh_data;
params.vash_data = vash_data;
params.ionet_data = ionet_data;
params.invnet_data = invnet_data;

%% Aggregate consumption (NIPA)
Cons_agg = [];
if exist('Real GDP Components.xls', 'file')
    Cons_agg = xlsread('Real GDP Components.xls', 1, 'C9:BU9')';
elseif exist('Data/Real GDP Components.xls', 'file')
    Cons_agg = xlsread('Data/Real GDP Components.xls', 1, 'C9:BU9')';
end

%% Empirical targets
empirical_targets = compute_empirical_targets(VA_raw, EMP_raw, InvRaw, VAn, Invn, GO47bea, VA47bea, Cons_agg);

%% Client rankings and labels
[client_indices, ranking] = compute_client_rankings(ionet_data, sector_indices, n_sectors);

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

%% Output
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
