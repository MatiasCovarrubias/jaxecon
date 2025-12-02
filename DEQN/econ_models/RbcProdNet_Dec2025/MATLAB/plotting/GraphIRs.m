function GraphIRs(irs_1, irs_2, irs_3, ax, N, labels, range_padding, opts, legend_labels)
% GRAPHIRS Plot impulse response figures for production network model
%
% This function creates 5 figures showing impulse responses:
%   1. Aggregate variables
%   2. Shocked sector inputs
%   3. Shocked sector outputs
%   4. Client sector inputs
%   5. Client sector outputs
%
% INPUTS:
%   irs_1          - Primary IRF data (cell array, one per sector)
%   irs_2          - Secondary IRF data (optional, can be empty)
%   irs_3          - Tertiary IRF data (optional, can be empty)
%   ax             - X-axis values (0:N-1)
%   N              - Number of periods to plot
%   labels         - Labels structure (struct or old cell format)
%   range_padding  - Y-axis padding as fraction of range
%   opts           - Options structure:
%                    - figures_folder: folder to save figures (default: 'output')
%                    - save_label: label for saved files (default: '')
%                    - save_figures: true/false to save figures (default: false)
%   legend_labels  - (Optional) Cell array of legend labels for series
%                    Default: {'Nonlinear', 'Log-linear', 'Cobb-Douglas'}

    tic;
    
    % Handle opts structure
    if ~isstruct(opts)
        % Backward compatibility: opts is save_label string
        save_label = opts;
        figures_folder = 'output';
        save_figures = false;
        if nargin >= 9 && ~iscell(legend_labels) && isscalar(legend_labels)
            save_figures = legend_labels;
            legend_labels = {};
        end
    else
        figures_folder = opts.figures_folder;
        save_label = opts.save_label;
        save_figures = opts.save_figures;
    end
    
    % Extract labels
    sector_indices = labels.sector_indices;
    sector_labels = labels.sector_labels;
    client_indices = labels.client_indices;
    client_labels = labels.client_labels;
    
    % Get LaTeX-safe labels for titles (with escaped underscores)
    if isfield(labels, 'sector_labels_latex')
        sector_labels_latex = labels.sector_labels_latex;
        client_labels_latex = labels.client_labels_latex;
    else
        sector_labels_latex = sector_labels;
        client_labels_latex = client_labels;
    end
    
    % Get filename-safe labels
    if isfield(labels, 'sector_labels_filename')
        sector_labels_filename = labels.sector_labels_filename;
    else
        sector_labels_filename = sector_labels;
    end
    
    % Determine plot mode based on inputs
    n_series = 1;
    if ~isempty(irs_2)
        n_series = 2;
    end
    if ~isempty(irs_3)
        n_series = 3;
    end
    
    % Set default legend labels if not provided
    if nargin < 9 || isempty(legend_labels) || ~iscell(legend_labels)
        legend_labels = get_default_legend_labels(n_series);
    end
    
    % Create figures
    local_fig_agg = figure('Position', [100, 100, 800, 800]);
    local_fig_sec_in = figure('Position', [100, 100, 800, 800]);
    local_fig_sec_out = figure('Position', [100, 100, 800, 800]);
    local_fig_client_in = figure('Position', [100, 100, 800, 800]);
    local_fig_client_out = figure('Position', [100, 100, 800, 800]);
    
    % Define line styles
    line_specs = get_line_specs(n_series);
    legend_entries = ['Steady State', legend_labels(1:n_series)];
    
    for idx = 1:numel(sector_indices)
        sector_idx = sector_indices(idx);
        if iscell(sector_labels_latex)
            sector_label_latex = sector_labels_latex{idx};
        else
            sector_label_latex = sector_labels_latex(idx);
        end
        client_idx = client_indices(idx);
        if iscell(client_labels_latex)
            client_label_latex = client_labels_latex{idx};
        else
            client_label_latex = client_labels_latex(idx);
        end
        
        title_sector = [char(sector_label_latex), ' shock'];
        
        % Get data for this sector
        d1 = irs_1{idx};
        d2 = get_data_or_empty(irs_2, idx);
        d3 = get_data_or_empty(irs_3, idx);
        
        % Determine time offset (some modes use offset)
        t_offset = get_time_offset(n_series);
        
        % ========== AGGREGATE PANEL ==========
        figure(local_fig_agg);
        plot_aggregate_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset);
        sgtitle(sprintf('%s, effect on \\textbf{Aggregate}', title_sector), ...
            'Interpreter', 'latex', 'Fontsize', 18);
        
        % ========== SECTOR INPUTS PANEL ==========
        figure(local_fig_sec_in);
        plot_sector_inputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset);
        sgtitle(sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, sector_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', 18);
        
        % ========== SECTOR OUTPUTS PANEL ==========
        figure(local_fig_sec_out);
        plot_sector_outputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset);
        sgtitle(sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, sector_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', 18);
        
        % ========== CLIENT INPUTS PANEL ==========
        figure(local_fig_client_in);
        plot_client_inputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset);
        sgtitle(sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, client_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', 18);
        
        % ========== CLIENT OUTPUTS PANEL ==========
        figure(local_fig_client_out);
        plot_client_outputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset);
        sgtitle(sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, client_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', 18);
    end
    
    % Save figures if requested
    if save_figures
        if iscell(sector_labels_filename)
            sec_label_str = sector_labels_filename{end};
        else
            sec_label_str = char(sector_labels_filename(end));
        end
        
        % Sanitize labels for filesystem safety
        save_label_safe = char(save_label);
        save_label_safe = regexprep(save_label_safe, '[^a-zA-Z0-9_\-]', '');
        sec_label_str = regexprep(sec_label_str, '[^a-zA-Z0-9_\-]', '');
        
        % Create figures folder if needed
        if ~exist(figures_folder, 'dir')
            mkdir(figures_folder);
        end
        
        % Build filename prefix
        if isempty(save_label_safe)
            prefix = '';
        else
            prefix = [save_label_safe, '_'];
        end
        
        saveas(local_fig_sec_in, fullfile(figures_folder, ['IR_IN_', prefix, sec_label_str, '.png']));
        saveas(local_fig_sec_out, fullfile(figures_folder, ['IR_OUT_', prefix, sec_label_str, '.png']));
        saveas(local_fig_agg, fullfile(figures_folder, ['IR_AGGR_', prefix, sec_label_str, '.png']));
        saveas(local_fig_client_in, fullfile(figures_folder, ['IR_CLIENT_IN_', prefix, sec_label_str, '.png']));
        saveas(local_fig_client_out, fullfile(figures_folder, ['IR_CLIENT_OUT_', prefix, sec_label_str, '.png']));
        
        fprintf('Figures saved to: %s\n', figures_folder);
    end
    
    disp(' *** FINISHED CREATING THE IRs ***');
    fprintf('It took %.4f seconds to generate and print the graphs.\n', toc);
end

%% ==================== Helper Functions ====================

function labels = get_default_legend_labels(n_series)
    if n_series == 1
        labels = {'Nonlinear'};
    elseif n_series == 2
        labels = {'Nonlinear', 'Log-linear'};
    else
        labels = {'Nonlinear', 'Log-linear', 'Cobb-Douglas'};
    end
end

function specs = get_line_specs(n_series)
    if n_series == 1
        specs = {{'k', 1.5}};
    elseif n_series == 2
        specs = {{'k', 1.5}, {'--r', 1.5}};
    else
        specs = {{'k', 1.5}, {'--r', 1.5}, {'-.b', 1.5}};
    end
end

function data = get_data_or_empty(irs, idx)
    if isempty(irs)
        data = [];
    else
        data = irs{idx};
    end
end

function t_offset = get_time_offset(n_series)
    if n_series == 3
        t_offset = 1;
    else
        t_offset = 0;
    end
end

function plot_subplot_multi(ax, baseline, d1, d2, d3, row, N, range_padding, specs, t_off)
    hold on;
    plot(ax, baseline * ones(N, 1), '--k', 'LineWidth', 1);
    
    t1 = 1 + t_off;
    t2 = N + t_off;
    
    plot(ax, d1(row, t1:t2), specs{1}{1}, 'LineWidth', specs{1}{2});
    if ~isempty(d2)
        plot(ax, d2(row, t1:t2), specs{2}{1}, 'LineWidth', specs{2}{2});
    end
    if ~isempty(d3)
        plot(ax, d3(row, t1:t2), specs{3}{1}, 'LineWidth', specs{3}{2});
    end
    hold off;
    
    YLim = get(gca, 'YLim');
    padding = range_padding * range(YLim);
    set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]);
    yaxis = get(gca, 'YAxis');
    yaxis.Exponent = 0;
    set(gca, 'Fontsize', 12);
end

%% ==================== Panel Plotting Functions ====================

function plot_aggregate_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off)
    n_cols = 2;
    n_rows = 2;
    
    subplot(n_rows, n_cols, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0);
    title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    legend(legend_entries, 'Location', 'southeast');
    
    subplot(n_rows, n_cols, 2);
    plot_subplot_multi(ax, 0, d1, d2, d3, 2, N, padding, specs, t_off);
    title('$C$', 'Interpreter', 'latex', 'Fontsize', 12);
    
    subplot(n_rows, n_cols, 3);
    plot_subplot_multi(ax, 0, d1, d2, d3, 3, N, padding, specs, t_off);
    title('$L$', 'Interpreter', 'latex', 'Fontsize', 12);
    
    subplot(n_rows, n_cols, 4);
    plot_subplot_multi(ax, 0, d1, d2, d3, 24, N, padding, specs, t_off);
    title('$Y$', 'Interpreter', 'latex', 'Fontsize', 12);
end

function plot_sector_inputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off)
    subplot(2, 3, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0);
    title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    legend(legend_entries, 'Location', 'southeast');
    
    subplot(2, 3, 2);
    plot_subplot_multi(ax, 0, d1, d2, d3, 8, N, padding, specs, t_off);
    title('$L_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    
    subplot(2, 3, 3);
    plot_subplot_multi(ax, 0, d1, d2, d3, 9, N, padding, specs, t_off);
    title('$I_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    
    subplot(2, 3, 4);
    plot_subplot_multi(ax, 0, d1, d2, d3, 10, N, padding, specs, t_off);
    title('$M_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    
    subplot(2, 3, 5);
    plot_subplot_multi(ax, 0, d1, d2, d3, 11, N, padding, specs, t_off);
    title('$Y_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
end

function plot_sector_outputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off)
    subplot(2, 3, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0);
    title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    legend(legend_entries, 'Location', 'southeast');
    
    subplot(2, 3, 2);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 5, N, padding, specs, t_off);
        title('$P_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 4, N, padding, specs, t_off);
        title('$C_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
    
    subplot(2, 3, 3);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 12, N, padding, specs, t_off);
        title('$Q_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 5, N, padding, specs, t_off);
        title('$P_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
    
    subplot(2, 3, 4);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 4, N, padding, specs, t_off);
        title('$C_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 7, N, padding, specs, t_off);
        title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
    
    subplot(2, 3, 5);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 7, N, padding, specs, t_off);
        title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 6, N, padding, specs, t_off);
        title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
    
    subplot(2, 3, 6);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 6, N, padding, specs, t_off);
        title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 12, N, padding, specs, t_off);
        title('$Q_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
end

function plot_client_inputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off)
    if n_series < 3
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 1, d1, d2, d3, 13, N, padding, specs, 0);
        title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        legend(legend_entries, 'Location', 'southeast');
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 18, N, padding, specs, t_off);
        title('$L_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 19, N, padding, specs, t_off);
        title('$I_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 20, N, padding, specs, t_off);
        title('$M_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 21, N, padding, specs, t_off);
        title('$Y_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 0, d1, d2, d3, 18, N, padding, specs, t_off);
        title('$L_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 19, N, padding, specs, t_off);
        title('$I_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 20, N, padding, specs, t_off);
        title('$M_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        legend(legend_entries, 'Location', 'southeast');
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 21, N, padding, specs, t_off);
        title('$Y_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 25, N, padding, specs, t_off);
        title('$P^m_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 6);
        plot_subplot_multi(ax, 0, d1, d2, d3, 26, N, padding, specs, t_off);
        title('$\hat{\gamma}_{ij}$ (Expend. Share)', 'Interpreter', 'latex', 'Fontsize', 12);
    end
end

function plot_client_outputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off)
    if n_series < 3
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 1, d1, d2, d3, 13, N, padding, specs, 0);
        title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        legend(legend_entries, 'Location', 'southeast');
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 14, N, padding, specs, t_off);
        title('$C_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 15, N, padding, specs, t_off);
        title('$P_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 17, N, padding, specs, t_off);
        title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 16, N, padding, specs, t_off);
        title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 6);
        plot_subplot_multi(ax, 0, d1, d2, d3, 22, N, padding, specs, t_off);
        title('$Q_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    else
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 0, d1, d2, d3, 15, N, padding, specs, t_off);
        title('$P_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 22, N, padding, specs, t_off);
        title('$Q_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 14, N, padding, specs, t_off);
        title('$C_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        legend(legend_entries, 'Location', 'southeast');
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 17, N, padding, specs, t_off);
        title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 16, N, padding, specs, t_off);
        title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
    end
end
