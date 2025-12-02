function labels = SectorLabel(indices)
% SECTORLABEL Returns BEA sector labels for given indices
%
% INPUTS:
%   indices - Vector of sector indices (1-37)
%
% OUTPUTS:
%   labels - Structure with multiple label formats:
%            - raw: Original labels (cell array)
%            - display: Labels with underscores as spaces (cell array)
%            - latex: Labels safe for LaTeX interpreter (cell array)
%            - filename: Labels safe for filenames (cell array)
%
% The 37 sectors correspond to BEA industry classifications.

    sector_names = {
        'Mining_Oil_and_Gas';    % 1
        'Utilities';              % 2
        'Construction';           % 3
        'Wood';                   % 4
        'Minerals';               % 5
        'Primary_Metals';         % 6
        'Fabricated_Metals';      % 7
        'Machinery';              % 8
        'Computers';              % 9
        'Electrical';             % 10
        'Vehicles';               % 11
        'Transport';              % 12
        'Furniture';              % 13
        'Misc_Mfg';               % 14
        'Food_Mfg';               % 15
        'Textile';                % 16
        'Apparel';                % 17
        'Paper';                  % 18
        'Printing';               % 19
        'Petroleum';              % 20
        'Chemical';               % 21
        'Plastics';               % 22
        'Wholesale_Trade';        % 23
        'Retail';                 % 24
        'Transp_and_Wareh';       % 25
        'Info';                   % 26
        'Finance';                % 27
        'Real_estate';            % 28
        'Prof_Tech';              % 29
        'Mgmt';                   % 30
        'Admin';                  % 31
        'Educ';                   % 32
        'Health';                 % 33
        'Arts';                   % 34
        'Accom';                  % 35
        'Food_Services';          % 36
        'Other';                  % 37
    };
    
    % Validate indices
    if any(indices < 1) || any(indices > 37)
        error('SectorLabel:InvalidIndex', 'Sector indices must be between 1 and 37');
    end
    
    raw_labels = sector_names(indices);
    n = numel(indices);
    
    display_labels = cell(n, 1);
    latex_labels = cell(n, 1);
    filename_labels = cell(n, 1);
    
    for i = 1:n
        raw = raw_labels{i};
        display_labels{i} = strrep(raw, '_', ' ');
        latex_labels{i} = strrep(raw, '_', '\_');
        filename_labels{i} = regexprep(raw, '[^a-zA-Z0-9]', '');
    end
    
    labels = struct();
    labels.raw = raw_labels;
    labels.display = display_labels;
    labels.latex = latex_labels;
    labels.filename = filename_labels;
end

