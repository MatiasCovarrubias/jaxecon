function validate_sector_indices(sector_indices, n_sectors, context)
% VALIDATE_SECTOR_INDICES Validate that sector indices are within valid range
%
% INPUTS:
%   sector_indices - Vector of sector indices to validate
%   n_sectors      - Maximum valid sector index (typically 37)
%   context        - String describing where validation is happening
%
% EXAMPLES:
%   validate_sector_indices([1, 20, 24], 37, 'main_IRs');
%   validate_sector_indices(1:37, 37, 'compute_all_sectors');

    if nargin < 3
        context = 'unknown function';
    end
    
    if isempty(sector_indices)
        error('validate_sector_indices:EmptyIndices', ...
            '[%s] sector_indices cannot be empty', context);
    end
    
    if any(sector_indices < 1)
        error('validate_sector_indices:InvalidLower', ...
            '[%s] sector_indices must be >= 1. Got minimum: %d', ...
            context, min(sector_indices));
    end
    
    if any(sector_indices > n_sectors)
        error('validate_sector_indices:InvalidUpper', ...
            '[%s] sector_indices must be <= %d. Got maximum: %d', ...
            context, n_sectors, max(sector_indices));
    end
    
    if any(floor(sector_indices) ~= sector_indices)
        error('validate_sector_indices:NonInteger', ...
            '[%s] sector_indices must be integers', context);
    end
end

