function validate_params(params, required_fields, context)
% VALIDATE_PARAMS Validate that required fields exist in params structure
%
% This function provides defensive input validation to catch configuration
% errors early with informative error messages.
%
% INPUTS:
%   params          - Structure to validate
%   required_fields - Cell array of required field names
%   context         - String describing where validation is happening (for error messages)
%
% EXAMPLES:
%   validate_params(params, {'n_sectors', 'beta', 'delta'}, 'calibrate_steady_state');
%   validate_params(params, {'Gamma_M', 'sigma_m'}, 'process_ir_data');

    if nargin < 3
        context = 'unknown function';
    end
    
    missing_fields = {};
    for i = 1:numel(required_fields)
        if ~isfield(params, required_fields{i})
            missing_fields{end+1} = required_fields{i}; %#ok<AGROW>
        end
    end
    
    if ~isempty(missing_fields)
        error('validate_params:MissingFields', ...
            '[%s] Missing required fields in params: %s', ...
            context, strjoin(missing_fields, ', '));
    end
end

