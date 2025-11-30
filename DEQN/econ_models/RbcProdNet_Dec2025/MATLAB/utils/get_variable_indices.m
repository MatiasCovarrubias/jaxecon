function indices = get_variable_indices(n_sectors)
% GET_VARIABLE_INDICES Returns standardized variable index ranges for the model
%
% This function provides a single source of truth for variable indexing
% in Dynare output and policy vectors. All other functions should use
% these indices rather than hardcoding offsets.
%
% INPUTS:
%   n_sectors - Number of sectors in the model (typically 37)
%
% OUTPUTS:
%   indices - Structure with index ranges for each variable type:
%
%   STATES (in Dynare ordering):
%     indices.k     - Capital: [1, n_sectors]
%     indices.a     - TFP: [n_sectors+1, 2*n_sectors]
%
%   POLICIES (in Dynare ordering):
%     indices.c     - Consumption: [2*n+1, 3*n]
%     indices.l     - Labor: [3*n+1, 4*n]
%     indices.pk    - Capital price: [4*n+1, 5*n]
%     indices.pm    - Intermediate price: [5*n+1, 6*n]
%     indices.m     - Intermediate input: [6*n+1, 7*n]
%     indices.mout  - Intermediate output: [7*n+1, 8*n]
%     indices.i     - Investment input: [8*n+1, 9*n]
%     indices.iout  - Investment output: [9*n+1, 10*n]
%     indices.p     - Price: [10*n+1, 11*n]
%     indices.q     - Tobin's Q: [11*n+1, 12*n]
%     indices.y     - Output: [12*n+1, 13*n]
%
%   AGGREGATES (scalar indices):
%     indices.cagg  - Aggregate consumption: 13*n+1
%     indices.lagg  - Aggregate labor: 13*n+2
%     indices.yagg  - Aggregate output: 13*n+3
%     indices.iagg  - Aggregate investment: 13*n+4
%     indices.magg  - Aggregate intermediates: 13*n+5
%     indices.V     - Value function: 13*n+6
%     indices.Vc    - Value function (consumption): 13*n+7
%
%   POLICIES_SS (in policies_ss vector ordering, 0-indexed blocks):
%     indices.ss_c     - Block 0: [1, n]
%     indices.ss_l     - Block 1: [n+1, 2*n]
%     indices.ss_pk    - Block 2: [2*n+1, 3*n]
%     indices.ss_pm    - Block 3: [3*n+1, 4*n]
%     indices.ss_m     - Block 4: [4*n+1, 5*n]
%     indices.ss_mout  - Block 5: [5*n+1, 6*n]
%     indices.ss_i     - Block 6: [6*n+1, 7*n]
%     indices.ss_iout  - Block 7: [7*n+1, 8*n]
%     indices.ss_p     - Block 8: [8*n+1, 9*n]
%     indices.ss_q     - Block 9: [9*n+1, 10*n]
%     indices.ss_y     - Block 10: [10*n+1, 11*n]

    n = n_sectors;
    indices = struct();
    
    %% Dynare variable indices (states + policies in Dynare ordering)
    % States
    indices.k = [1, n];
    indices.a = [n+1, 2*n];
    
    % Policies (Dynare ordering)
    indices.c = [2*n+1, 3*n];
    indices.l = [3*n+1, 4*n];
    indices.pk = [4*n+1, 5*n];
    indices.pm = [5*n+1, 6*n];
    indices.m = [6*n+1, 7*n];
    indices.mout = [7*n+1, 8*n];
    indices.i = [8*n+1, 9*n];
    indices.iout = [9*n+1, 10*n];
    indices.p = [10*n+1, 11*n];
    indices.q = [11*n+1, 12*n];
    indices.y = [12*n+1, 13*n];
    
    % Aggregates (scalar indices)
    indices.cagg = 13*n + 1;
    indices.lagg = 13*n + 2;
    indices.yagg = 13*n + 3;
    indices.iagg = 13*n + 4;
    indices.magg = 13*n + 5;
    indices.V = 13*n + 6;
    indices.Vc = 13*n + 7;
    
    %% Steady-state policy vector indices (policies_ss ordering)
    % These match the block structure: block_k starts at k*n + 1
    indices.ss_c = [1, n];           % Block 0
    indices.ss_l = [n+1, 2*n];       % Block 1
    indices.ss_pk = [2*n+1, 3*n];    % Block 2
    indices.ss_pm = [3*n+1, 4*n];    % Block 3
    indices.ss_m = [4*n+1, 5*n];     % Block 4
    indices.ss_mout = [5*n+1, 6*n];  % Block 5
    indices.ss_i = [6*n+1, 7*n];     % Block 6
    indices.ss_iout = [7*n+1, 8*n];  % Block 7
    indices.ss_p = [8*n+1, 9*n];     % Block 8
    indices.ss_q = [9*n+1, 10*n];    % Block 9
    indices.ss_y = [10*n+1, 11*n];   % Block 10
    
    % Steady-state aggregates
    indices.ss_cagg = 11*n + 1;
    indices.ss_lagg = 11*n + 2;
    indices.ss_yagg = 11*n + 3;
    indices.ss_iagg = 11*n + 4;
    indices.ss_magg = 11*n + 5;
    indices.ss_V = 11*n + 6;
    
    %% Helper: number of sectors
    indices.n_sectors = n;
    
    %% Helper functions for getting specific sector indices
    % These return the index for sector j within a variable block
    indices.get_dynare_idx = @(var_range, j) var_range(1) + j - 1;
    indices.get_ss_idx = @(ss_range, j) ss_range(1) + j - 1;
end

