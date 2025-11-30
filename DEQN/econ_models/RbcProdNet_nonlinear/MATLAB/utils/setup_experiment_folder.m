function exp_paths = setup_experiment_folder(exp_name, base_dir)
% SETUP_EXPERIMENT_FOLDER Create and return paths for experiment folder structure
%
% INPUTS:
%   exp_name - Name of the experiment (used as folder name)
%   base_dir - Base directory (default: current directory)
%
% OUTPUTS:
%   exp_paths - Structure with paths:
%               - root: experiments folder
%               - experiment: this experiment's folder
%               - temp: temp subfolder
%               - figures: figures subfolder
%
% FOLDER STRUCTURE:
%   experiments/
%       <exp_name>/
%           temp/           - temporary/intermediate files
%           figures/        - saved figures
%           ModelData.mat   - main results file

if nargin < 2 || isempty(base_dir)
    base_dir = pwd;
end

% Sanitize experiment name for filesystem safety
exp_name_safe = regexprep(char(exp_name), '[^a-zA-Z0-9_\-]', '_');
exp_name_safe = regexprep(exp_name_safe, '_+', '_');
exp_name_safe = regexprep(exp_name_safe, '^_|_$', '');

% Build paths
exp_paths = struct();
exp_paths.root = fullfile(base_dir, 'experiments');
exp_paths.experiment = fullfile(exp_paths.root, exp_name_safe);
exp_paths.temp = fullfile(exp_paths.experiment, 'temp');
exp_paths.figures = fullfile(exp_paths.experiment, 'figures');

% Create directories if they don't exist
dirs_to_create = {exp_paths.root, exp_paths.experiment, exp_paths.temp, exp_paths.figures};
for i = 1:numel(dirs_to_create)
    if ~exist(dirs_to_create{i}, 'dir')
        mkdir(dirs_to_create{i});
        fprintf('Created folder: %s\n', dirs_to_create{i});
    end
end

end

