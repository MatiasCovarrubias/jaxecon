function Results = process_sector_irs(DynareResults, params, ModData, labels, opts)
% PROCESS_SECTOR_IRS Process IRFs for specified sectors
%
% This is the unified function for processing impulse responses. It handles
% both single/few sectors and all-sectors analysis with a single code path.
%
% INPUTS:
%   DynareResults  - Structure from run_dynare_analysis containing IRF data
%   params         - Model parameters structure
%   ModData        - Steady state data structure
%   labels         - Labels struct with fields:
%                    - sector_indices, sector_labels, client_indices, client_labels
%   opts           - Options structure:
%                    - plot_graphs: Generate plots (default: true for <5 sectors)
%                    - save_graphs: Save plots to files (default: false)
%                    - save_intermediate: Save intermediate results (default: false)
%                    - save_interval: How often to save (default: 5)
%                    - exp_paths: Experiment paths structure from setup_experiment_folder
%                    - save_label: Label for files (default: '')
%                    - range_padding: Y-axis padding for plots (default: 0.1)
%                    - ir_plot_length: Number of periods for plots (default: 60)
%                    - shock_description: Description of shock for plot titles (default: '')
%                    - shock_sign: +1 for positive shock, -1 for negative (default: auto-detect)
%
% OUTPUTS:
%   Results - Structure containing:
%             - IRFs: Cell array {sector_idx} with IRSLoglin and IRSDeterm
%             - Statistics: Peak values, half-lives, amplifications per sector

%% Input validation
validate_params(params, {'n_sectors'}, 'process_sector_irs');

%% Set defaults
if nargin < 5
    opts = struct();
end

sector_indices = labels.sector_indices;
n_analyzed = numel(sector_indices);
n_sectors = params.n_sectors;

validate_sector_indices(sector_indices, n_sectors, 'process_sector_irs');

% Default: plot graphs only for small numbers of sectors
default_plot = (n_analyzed <= 5);

opts = set_default(opts, 'plot_graphs', default_plot);
opts = set_default(opts, 'save_graphs', false);
opts = set_default(opts, 'save_intermediate', false);
opts = set_default(opts, 'save_interval', 5);
opts = set_default(opts, 'exp_paths', struct('temp', 'output', 'figures', 'output'));
opts = set_default(opts, 'save_label', '');
opts = set_default(opts, 'range_padding', 0.1);
opts = set_default(opts, 'ir_plot_length', 60);
opts = set_default(opts, 'shock_description', '');
opts = set_default(opts, 'shock_sign', []);  % Empty = auto-detect

%% Extract steady state data
policies_ss = ModData.policies_ss;
k_ss = ModData.endostates_ss;
Cagg_ss = DynareResults.Cagg_ss;
Lagg_ss = DynareResults.Lagg_ss;
steady_state = DynareResults.steady_state;

%% Initialize storage
fprintf('=== PROCESSING IRFs FOR %d SECTORS ===\n', n_analyzed);
if ~isempty(opts.shock_description)
    fprintf('    Shock: %s\n', opts.shock_description);
end

IRFs = cell(n_analyzed, 1);

% Statistics arrays (indexed by position in sector_indices, not sector number)
Stats = struct();
Stats.sector_indices = sector_indices;
Stats.peak_values_loglin = zeros(n_analyzed, 1);
Stats.peak_values_determ = zeros(n_analyzed, 1);
Stats.peak_periods_loglin = zeros(n_analyzed, 1);
Stats.peak_periods_determ = zeros(n_analyzed, 1);
Stats.half_lives_loglin = zeros(n_analyzed, 1);
Stats.half_lives_determ = zeros(n_analyzed, 1);
Stats.amplifications = zeros(n_analyzed, 1);

%% Process each sector
for idx = 1:n_analyzed
    sector_idx = sector_indices(idx);
    client_idx = labels.client_indices(idx);
    
    fprintf('Processing sector %d (%d/%d)\n', sector_idx, idx, n_analyzed);
    
    % Process log-linear IRF
    dynare_simul_loglin = DynareResults.IRSLoglin_raw{idx};
    IRSLoglin = process_ir_data(dynare_simul_loglin, sector_idx, client_idx, params, ...
        steady_state, n_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    
    % Process deterministic IRF
    dynare_simul_determ = DynareResults.IRSDeterm_raw{idx};
    IRSDeterm = process_ir_data(dynare_simul_determ, sector_idx, client_idx, params, ...
        steady_state, n_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    
    % Store IRFs
    IRFs{idx}.sector_idx = sector_idx;
    IRFs{idx}.client_idx = client_idx;
    IRFs{idx}.IRSLoglin = IRSLoglin;
    IRFs{idx}.IRSDeterm = IRSDeterm;
    
    % Compute statistics (row 2 is aggregate C)
    T_stats = min(100, size(IRSLoglin, 2));
    C_loglin = IRSLoglin(2, 1:T_stats);
    C_determ = IRSDeterm(2, 1:T_stats);
    
    % Determine shock sign: auto-detect from TFP response if not provided
    % Row 1 is A_ir (TFP level), A > 1 means positive shock, A < 1 means negative
    if isempty(opts.shock_sign)
        A_initial = IRSLoglin(1, 1);
        if A_initial > 1
            shock_sign = 1;   % Positive shock
        else
            shock_sign = -1;  % Negative shock
        end
    else
        shock_sign = opts.shock_sign;
    end
    
    % Calculate peak statistics
    % For negative shock: consumption drops, so we look at max drop (peak of -C)
    % For positive shock: consumption rises, so we look at max rise (peak of C)
    [pv_loglin, pp_loglin, hl_loglin] = calculatePeaksAndHalfLives(shock_sign * C_loglin);
    [pv_determ, pp_determ, hl_determ] = calculatePeaksAndHalfLives(shock_sign * C_determ);
    
    % Store absolute peak values (positive number = magnitude of response)
    Stats.peak_values_loglin(idx) = abs(pv_loglin);
    Stats.peak_values_determ(idx) = abs(pv_determ);
    Stats.peak_periods_loglin(idx) = pp_loglin;
    Stats.peak_periods_determ(idx) = pp_determ;
    Stats.half_lives_loglin(idx) = hl_loglin;
    Stats.half_lives_determ(idx) = hl_determ;
    Stats.amplifications(idx) = abs(pv_determ) - abs(pv_loglin);
    
    % Print with sign indication
    sign_str = '';
    if shock_sign > 0
        sign_str = ' (+)';
    else
        sign_str = ' (-)';
    end
    fprintf('  peak=%.4f (loglin), %.4f (determ)%s, amplification=%.4f, half-life=%d/%d\n', ...
        abs(pv_loglin), abs(pv_determ), sign_str, Stats.amplifications(idx), hl_loglin, hl_determ);
    
    % Save intermediate results if requested
    if opts.save_intermediate && mod(idx, opts.save_interval) == 0
        intermediate_file = fullfile(opts.exp_paths.temp, ...
            ['IRFs_Intermediate_' opts.save_label '.mat']);
        save(intermediate_file, 'IRFs', 'Stats');
        fprintf('  Saved intermediate results: %s\n', intermediate_file);
    end
end

%% Plot graphs if requested
if opts.plot_graphs
    fprintf('Generating plots...\n');
    
    % Build cell arrays for GraphIRs (expects cell format)
    IRSLoglin_cells = cell(n_analyzed, 1);
    IRSDeterm_cells = cell(n_analyzed, 1);
    for idx = 1:n_analyzed
        IRSLoglin_cells{idx} = IRFs{idx}.IRSLoglin;
        IRSDeterm_cells{idx} = IRFs{idx}.IRSDeterm;
    end
    
    ax = 0:(opts.ir_plot_length - 1);
    graph_opts = struct();
    graph_opts.figures_folder = opts.exp_paths.figures;
    graph_opts.save_label = opts.save_label;
    graph_opts.save_figures = opts.save_graphs;
    graph_opts.shock_description = opts.shock_description;
    
    GraphIRs(IRSDeterm_cells, IRSLoglin_cells, [], ax, opts.ir_plot_length, ...
        labels, opts.range_padding, graph_opts);
end

%% Print summary
fprintf('\n=== IRF STATISTICS SUMMARY ===\n');
fprintf('Sectors analyzed: %d\n', n_analyzed);
if ~isempty(opts.shock_description)
    fprintf('Shock: %s\n', opts.shock_description);
end
fprintf('                    Log-Linear    Deterministic\n');
fprintf('Avg peak value:     %.4f        %.4f\n', ...
    mean(Stats.peak_values_loglin), mean(Stats.peak_values_determ));
fprintf('Avg peak period:    %.1f          %.1f\n', ...
    mean(Stats.peak_periods_loglin), mean(Stats.peak_periods_determ));
fprintf('Avg half-life:      %.1f          %.1f\n', ...
    mean(Stats.half_lives_loglin), mean(Stats.half_lives_determ));
fprintf('Avg amplification:  %.4f\n', mean(Stats.amplifications));

%% Build output structure
Results = struct();
Results.IRFs = IRFs;
Results.Statistics = Stats;
Results.labels = labels;
Results.shock_description = opts.shock_description;

end
