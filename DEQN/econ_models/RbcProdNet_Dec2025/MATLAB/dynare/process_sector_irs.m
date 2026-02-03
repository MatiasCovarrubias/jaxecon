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

% Check which IRF types are available
has_1storder = isfield(DynareResults, 'IRSFirstOrder_raw') && ~isempty(DynareResults.IRSFirstOrder_raw);
has_2ndorder = isfield(DynareResults, 'IRSSecondOrder_raw') && ~isempty(DynareResults.IRSSecondOrder_raw);
has_determ = isfield(DynareResults, 'IRSPerfectForesight_raw') && ~isempty(DynareResults.IRSPerfectForesight_raw);

% Statistics arrays (indexed by position in sector_indices, not sector number)
Stats = struct();
Stats.sector_indices = sector_indices;
Stats.peak_values_firstorder = zeros(n_analyzed, 1);
Stats.peak_values_secondorder = zeros(n_analyzed, 1);
Stats.peak_values_pf = zeros(n_analyzed, 1);
Stats.peak_periods_firstorder = zeros(n_analyzed, 1);
Stats.peak_periods_secondorder = zeros(n_analyzed, 1);
Stats.peak_periods_pf = zeros(n_analyzed, 1);
Stats.half_lives_firstorder = zeros(n_analyzed, 1);
Stats.half_lives_secondorder = zeros(n_analyzed, 1);
Stats.half_lives_pf = zeros(n_analyzed, 1);
Stats.amplifications = zeros(n_analyzed, 1);  % PF vs 1st order
Stats.amplifications_2nd = zeros(n_analyzed, 1);  % 2nd order vs 1st order
Stats.amplifications_rel = zeros(n_analyzed, 1);  % Relative amplification (%)

%% Process each sector
for idx = 1:n_analyzed
    sector_idx = sector_indices(idx);
    client_idx = labels.client_indices(idx);
    
    fprintf('Processing sector %d (%d/%d)\n', sector_idx, idx, n_analyzed);
    
    % Process first-order IRF
    if has_1storder
        dynare_simul_1st = DynareResults.IRSFirstOrder_raw{idx};
        IRS1stOrder = process_ir_data(dynare_simul_1st, sector_idx, client_idx, params, ...
            steady_state, n_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    else
        IRS1stOrder = [];
    end
    
    % Process second-order IRF
    if has_2ndorder
        dynare_simul_2nd = DynareResults.IRSSecondOrder_raw{idx};
        IRS2ndOrder = process_ir_data(dynare_simul_2nd, sector_idx, client_idx, params, ...
            steady_state, n_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    else
        IRS2ndOrder = [];
    end
    
    % Process perfect foresight IRF
    if has_determ
        dynare_simul_pf = DynareResults.IRSPerfectForesight_raw{idx};
        IRSPerfForesight = process_ir_data(dynare_simul_pf, sector_idx, client_idx, params, ...
            steady_state, n_sectors, k_ss, Cagg_ss, Lagg_ss, policies_ss);
    else
        IRSPerfForesight = [];
    end
    
    % Store IRFs
    IRFs{idx}.sector_idx = sector_idx;
    IRFs{idx}.client_idx = client_idx;
    IRFs{idx}.IRSFirstOrder = IRS1stOrder;        % First-order (linear)
    IRFs{idx}.IRSSecondOrder = IRS2ndOrder;       % Second-order (quadratic)
    IRFs{idx}.IRSPerfectForesight = IRSPerfForesight; % Perfect foresight (nonlinear)
    
    % Compute statistics (row 2 is aggregate C)
    % Use first-order as reference; fall back to second-order or pf
    if has_1storder
        T_stats = min(100, size(IRS1stOrder, 2));
        C_1st = IRS1stOrder(2, 1:T_stats);
    else
        C_1st = [];
    end
    if has_2ndorder
        T_stats = min(100, size(IRS2ndOrder, 2));
        C_2nd = IRS2ndOrder(2, 1:T_stats);
    else
        C_2nd = [];
    end
    if has_determ
        T_stats = min(100, size(IRSPerfForesight, 2));
        C_pf = IRSPerfForesight(2, 1:T_stats);
    else
        C_pf = [];
    end
    
    % Determine shock sign: auto-detect from TFP response if not provided
    % Row 1 is A_ir (TFP level), A > 1 means positive shock, A < 1 means negative
    if isempty(opts.shock_sign)
        % Use first available IRF to determine sign
        if has_1storder
            A_initial = IRS1stOrder(1, 1);
        elseif has_2ndorder
            A_initial = IRS2ndOrder(1, 1);
        else
            A_initial = IRSPerfForesight(1, 1);
        end
        if A_initial > 1
            shock_sign = 1;   % Positive shock
        else
            shock_sign = -1;  % Negative shock
        end
    else
        shock_sign = opts.shock_sign;
    end
    
    % Calculate peak statistics for each available IRF type
    % For negative shock: consumption drops, so we look at max drop (peak of -C)
    % For positive shock: consumption rises, so we look at max rise (peak of C)
    
    % First-order stats
    if has_1storder
        [pv_1st, pp_1st, hl_1st] = calculatePeaksAndHalfLives(shock_sign * C_1st);
        Stats.peak_values_firstorder(idx) = abs(pv_1st);
        Stats.peak_periods_firstorder(idx) = pp_1st;
        Stats.half_lives_firstorder(idx) = hl_1st;
    else
        pv_1st = 0; pp_1st = 0; hl_1st = 0;
    end
    
    % Second-order stats
    if has_2ndorder
        [pv_2nd, pp_2nd, hl_2nd] = calculatePeaksAndHalfLives(shock_sign * C_2nd);
        Stats.peak_values_secondorder(idx) = abs(pv_2nd);
        Stats.peak_periods_secondorder(idx) = pp_2nd;
        Stats.half_lives_secondorder(idx) = hl_2nd;
    else
        pv_2nd = 0; pp_2nd = 0; hl_2nd = 0;
    end
    
    % Perfect foresight stats
    if has_determ
        [pv_pf, pp_pf, hl_pf] = calculatePeaksAndHalfLives(shock_sign * C_pf);
        Stats.peak_values_pf(idx) = abs(pv_pf);
        Stats.peak_periods_pf(idx) = pp_pf;
        Stats.half_lives_pf(idx) = hl_pf;
    else
        pv_pf = 0; pp_pf = 0; hl_pf = 0;
    end
    
    % Amplifications (PF vs 1st order, 2nd order vs 1st order)
    Stats.amplifications(idx) = abs(pv_pf) - abs(pv_1st);
    Stats.amplifications_2nd(idx) = abs(pv_2nd) - abs(pv_1st);
    if abs(pv_1st) > 1e-10
        Stats.amplifications_rel(idx) = (abs(pv_pf) / abs(pv_1st) - 1) * 100;  % Relative %
    else
        Stats.amplifications_rel(idx) = 0;
    end
    
    % Print with sign indication
    sign_str = '';
    if shock_sign > 0
        sign_str = ' (+)';
    else
        sign_str = ' (-)';
    end
    if has_2ndorder
        fprintf('  peak: 1st=%.4f, 2nd=%.4f, PF=%.4f%s | amplif: 2nd=%.4f, PF=%.2f%%\n', ...
            abs(pv_1st), abs(pv_2nd), abs(pv_pf), sign_str, ...
            Stats.amplifications_2nd(idx), Stats.amplifications_rel(idx));
    else
        fprintf('  peak=%.4f (1st), %.4f (PF)%s, amplification=%.2f%%, half-life=%d/%d\n', ...
            abs(pv_1st), abs(pv_pf), sign_str, Stats.amplifications_rel(idx), hl_1st, hl_pf);
    end
    
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
    
    ax = 0:(opts.ir_plot_length - 1);
    
    % Plot each sector separately to avoid overwriting
    for idx = 1:n_analyzed
        % Create single-sector data cells
        IRS1stOrder_single = {IRFs{idx}.IRSFirstOrder};
        IRS2ndOrder_single = {IRFs{idx}.IRSSecondOrder};
        IRSPerfForesight_single = {IRFs{idx}.IRSPerfectForesight};
        
        % Create single-sector labels structure
        labels_single = struct();
        labels_single.sector_indices = sector_indices(idx);
        labels_single.client_indices = labels.client_indices(idx);
        
        % Handle cell vs array label formats
        if iscell(labels.sector_labels)
            labels_single.sector_labels = labels.sector_labels(idx);
            labels_single.client_labels = labels.client_labels(idx);
        else
            labels_single.sector_labels = labels.sector_labels(idx);
            labels_single.client_labels = labels.client_labels(idx);
        end
        
        if isfield(labels, 'sector_labels_latex')
            if iscell(labels.sector_labels_latex)
                labels_single.sector_labels_latex = labels.sector_labels_latex(idx);
                labels_single.client_labels_latex = labels.client_labels_latex(idx);
            else
                labels_single.sector_labels_latex = labels.sector_labels_latex(idx);
                labels_single.client_labels_latex = labels.client_labels_latex(idx);
            end
        end
        
        if isfield(labels, 'sector_labels_filename')
            if iscell(labels.sector_labels_filename)
                labels_single.sector_labels_filename = labels.sector_labels_filename(idx);
            else
                labels_single.sector_labels_filename = labels.sector_labels_filename(idx);
            end
        end
        
        % Build graph options for this sector
        graph_opts = struct();
        graph_opts.figures_folder = opts.exp_paths.figures;
        graph_opts.save_figures = opts.save_graphs;
        graph_opts.shock_description = opts.shock_description;
        
        % Include sector index in save_label to distinguish files
        if ~isempty(opts.save_label)
            graph_opts.save_label = sprintf('%s_sec%d', opts.save_label, sector_indices(idx));
        else
            graph_opts.save_label = sprintf('sec%d', sector_indices(idx));
        end
        
        fprintf('  Plotting sector %d (%d/%d)...\n', sector_indices(idx), idx, n_analyzed);
        
        % Determine which IRFs to plot (order: PF, 1st, 2nd)
        % GraphIRs expects: irs_1 (primary), irs_2 (secondary), irs_3 (tertiary)
        if has_determ && has_1storder && has_2ndorder
            % All three available: PF (solid), 1st (dashed), 2nd (dash-dot)
            GraphIRs(IRSPerfForesight_single, IRS1stOrder_single, IRS2ndOrder_single, ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts);
        elseif has_determ && has_1storder
            % PF and 1st order only
            GraphIRs(IRSPerfForesight_single, IRS1stOrder_single, [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts);
        elseif has_determ && has_2ndorder
            % PF and 2nd order only
            GraphIRs(IRSPerfForesight_single, IRS2ndOrder_single, [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts, {'Perfect Foresight', 'Second-Order'});
        elseif has_1storder && has_2ndorder
            % 1st and 2nd order only (no PF)
            GraphIRs(IRS1stOrder_single, IRS2ndOrder_single, [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts, {'First-Order', 'Second-Order'});
        elseif has_determ
            % Only PF
            GraphIRs(IRSPerfForesight_single, [], [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts, {'Perfect Foresight'});
        elseif has_1storder
            % Only 1st order
            GraphIRs(IRS1stOrder_single, [], [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts, {'First-Order'});
        elseif has_2ndorder
            % Only 2nd order
            GraphIRs(IRS2ndOrder_single, [], [], ax, opts.ir_plot_length, ...
                labels_single, opts.range_padding, graph_opts, {'Second-Order'});
        end
    end
end

%% Print summary
fprintf('\n=== IRF STATISTICS SUMMARY ===\n');
fprintf('Sectors analyzed: %d\n', n_analyzed);
if ~isempty(opts.shock_description)
    fprintf('Shock: %s\n', opts.shock_description);
end
if has_2ndorder
    fprintf('                    First-Order   Second-Order  Perfect Foresight\n');
    fprintf('Avg peak value:     %.4f        %.4f        %.4f\n', ...
        mean(Stats.peak_values_firstorder), mean(Stats.peak_values_secondorder), mean(Stats.peak_values_pf));
    fprintf('Avg peak period:    %.1f           %.1f           %.1f\n', ...
        mean(Stats.peak_periods_firstorder), mean(Stats.peak_periods_secondorder), mean(Stats.peak_periods_pf));
    fprintf('Avg half-life:      %.1f           %.1f           %.1f\n', ...
        mean(Stats.half_lives_firstorder), mean(Stats.half_lives_secondorder), mean(Stats.half_lives_pf));
    fprintf('Amplification (2nd vs 1st): %.4f\n', mean(Stats.amplifications_2nd));
    fprintf('Amplification (PF vs 1st):  %.4f (%.1f%%)\n', mean(Stats.amplifications), mean(Stats.amplifications_rel));
else
    fprintf('                    First-Order   Perfect Foresight\n');
    fprintf('Avg peak value:     %.4f        %.4f\n', ...
        mean(Stats.peak_values_firstorder), mean(Stats.peak_values_pf));
    fprintf('Avg peak period:    %.1f           %.1f\n', ...
        mean(Stats.peak_periods_firstorder), mean(Stats.peak_periods_pf));
    fprintf('Avg half-life:      %.1f           %.1f\n', ...
        mean(Stats.half_lives_firstorder), mean(Stats.half_lives_pf));
    fprintf('Amplification (PF vs 1st): %.4f (%.1f%%)\n', mean(Stats.amplifications), mean(Stats.amplifications_rel));
end

%% Build output structure
Results = struct();
Results.IRFs = IRFs;
Results.Statistics = Stats;
Results.labels = labels;
Results.shock_description = opts.shock_description;

end
