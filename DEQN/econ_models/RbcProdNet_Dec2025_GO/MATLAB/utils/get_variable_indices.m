function idx = get_variable_indices(n)
% GET_VARIABLE_INDICES Returns variable index ranges for the model
%
% Single source of truth for variable indexing. Uses Dynare ordering
% (states first, then policies) as the canonical form.
%
% INPUTS:
%   n - Number of sectors (typically 37)
%
% OUTPUTS:
%   idx - Structure with index ranges:
%
%   STATES:
%     idx.k     - Capital: [1, n]
%     idx.a     - TFP: [n+1, 2*n]
%
%   POLICIES (11 sectoral blocks + 5 aggregates):
%     idx.c     - Consumption: [2*n+1, 3*n]
%     idx.l     - Labor: [3*n+1, 4*n]
%     idx.pk    - Capital price: [4*n+1, 5*n]
%     idx.pm    - Intermediate price: [5*n+1, 6*n]
%     idx.m     - Intermediate input: [6*n+1, 7*n]
%     idx.mout  - Intermediate output: [7*n+1, 8*n]
%     idx.i     - Investment: [8*n+1, 9*n]
%     idx.iout  - Investment output: [9*n+1, 10*n]
%     idx.p     - Price: [10*n+1, 11*n]
%     idx.q     - Gross output: [11*n+1, 12*n]
%     idx.y     - Value added: [12*n+1, 13*n]
%
%   AGGREGATES:
%     idx.cagg  - Aggregate consumption: 13*n+1
%     idx.lagg  - Aggregate labor: 13*n+2
%     idx.yagg  - Aggregate output: 13*n+3
%     idx.iagg  - Aggregate investment: 13*n+4
%     idx.magg  - Aggregate intermediates: 13*n+5
%
%   OFFSET for policies_ss:
%     idx.ss_offset = 2*n (subtract from Dynare index to get policies_ss index)
%
%   Example: To get y from policies_ss:
%     policies_ss(idx.y(1) - idx.ss_offset : idx.y(2) - idx.ss_offset)

    idx = struct();
    
    % States (Dynare indices 1 to 2n)
    idx.k = [1, n];
    idx.a = [n+1, 2*n];
    
    % Policies (Dynare indices 2n+1 onwards)
    idx.c = [2*n+1, 3*n];
    idx.l = [3*n+1, 4*n];
    idx.pk = [4*n+1, 5*n];
    idx.pm = [5*n+1, 6*n];
    idx.m = [6*n+1, 7*n];
    idx.mout = [7*n+1, 8*n];
    idx.i = [8*n+1, 9*n];
    idx.iout = [9*n+1, 10*n];
    idx.p = [10*n+1, 11*n];
    idx.q = [11*n+1, 12*n];
    idx.y = [12*n+1, 13*n];
    
    % Aggregates
    idx.cagg = 13*n + 1;
    idx.lagg = 13*n + 2;
    idx.yagg = 13*n + 3;
    idx.iagg = 13*n + 4;
    idx.magg = 13*n + 5;
    
    % Offset: subtract this from Dynare indices to get policies_ss indices
    idx.ss_offset = 2*n;
    
    % Number of sectors (for convenience)
    idx.n = n;
    
    % Total dimensions
    idx.n_states = 2*n;
    idx.n_policies = 11*n + 5;  % policies_ss length
    idx.n_dynare = 13*n + 5;    % Dynare simulation length
end
