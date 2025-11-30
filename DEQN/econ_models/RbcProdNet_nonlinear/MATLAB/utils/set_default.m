function opts = set_default(opts, field, default)
% SET_DEFAULT Set default value for a struct field if not already defined
%
% INPUTS:
%   opts    - Structure to modify
%   field   - Field name (string)
%   default - Default value to use if field doesn't exist
%
% OUTPUTS:
%   opts - Modified structure with field set to default if it was missing
%
% EXAMPLE:
%   opts = set_default(opts, 'verbose', true);
%   opts = set_default(opts, 'max_iter', 1000);

    if ~isfield(opts, field)
        opts.(field) = default;
    end
end

