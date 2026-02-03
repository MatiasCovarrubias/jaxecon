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
%                    - shock_description: string describing the shock (default: '')
%   legend_labels  - (Optional) Cell array of legend labels for series
%                    Default: {'Nonlinear', 'Log-linear', 'Cobb-Douglas'}

    tic;
    
    %% Handle opts structure
    if ~isstruct(opts)
        % Backward compatibility: opts is save_label string
        save_label = opts;
        figures_folder = 'output';
        save_figures = false;
        shock_description = '';
        if nargin >= 9 && ~iscell(legend_labels) && isscalar(legend_labels)
            save_figures = legend_labels;
            legend_labels = {};
        end
    else
        figures_folder = get_field_or_default(opts, 'figures_folder', 'output');
        save_label = get_field_or_default(opts, 'save_label', '');
        save_figures = get_field_or_default(opts, 'save_figures', false);
        shock_description = get_field_or_default(opts, 'shock_description', '');
    end
    
    %% Extract labels
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
    
    %% Determine plot mode based on inputs
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
    
    %% Setup styling
    style = get_plot_style();
    
    %% Create figures with improved sizing
    fig_pos = [100, 100, 900, 700];
    local_fig_agg = figure('Position', fig_pos, 'Color', 'w');
    local_fig_sec_in = figure('Position', fig_pos, 'Color', 'w');
    local_fig_sec_out = figure('Position', fig_pos, 'Color', 'w');
    local_fig_client_in = figure('Position', fig_pos, 'Color', 'w');
    local_fig_client_out = figure('Position', fig_pos, 'Color', 'w');
    
    %% Define line styles
    line_specs = get_line_specs(n_series, style);
    legend_entries = ['Steady State', legend_labels(1:n_series)];
    
    %% Build shock info for title
    if ~isempty(shock_description)
        % Escape special LaTeX characters in shock description
        shock_desc_safe = escape_latex(shock_description);
        shock_str = sprintf(' [%s]', shock_desc_safe);
    else
        shock_str = '';
    end
    
    %% Plot each sector
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
        plot_aggregate_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset, style);
        sgtitle(sprintf('%s%s, effect on \\textbf{Aggregate}', title_sector, shock_str), ...
            'Interpreter', 'latex', 'Fontsize', style.title_fontsize);
        
        % ========== SECTOR INPUTS PANEL ==========
        figure(local_fig_sec_in);
        plot_sector_inputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset, style);
        sgtitle(sprintf('%s%s, effect on \\textbf{%s} (INPUTS)', title_sector, shock_str, sector_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', style.title_fontsize);
        
        % ========== SECTOR OUTPUTS PANEL ==========
        figure(local_fig_sec_out);
        plot_sector_outputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset, style);
        sgtitle(sprintf('%s%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, shock_str, sector_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', style.title_fontsize);
        
        % ========== CLIENT INPUTS PANEL ==========
        figure(local_fig_client_in);
        plot_client_inputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset, style);
        sgtitle(sprintf('%s%s, effect on \\textbf{%s} (INPUTS)', title_sector, shock_str, client_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', style.title_fontsize);
        
        % ========== CLIENT OUTPUTS PANEL ==========
        figure(local_fig_client_out);
        plot_client_outputs_panel(ax, N, d1, d2, d3, n_series, range_padding, line_specs, legend_entries, t_offset, style);
        sgtitle(sprintf('%s%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, shock_str, client_label_latex), ...
            'Interpreter', 'latex', 'Fontsize', style.title_fontsize);
    end
    
    %% Save figures if requested
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
        
        % Save as PNG with higher resolution
        print(local_fig_sec_in, fullfile(figures_folder, ['IR_IN_', prefix, sec_label_str, '.png']), '-dpng', '-r150');
        print(local_fig_sec_out, fullfile(figures_folder, ['IR_OUT_', prefix, sec_label_str, '.png']), '-dpng', '-r150');
        print(local_fig_agg, fullfile(figures_folder, ['IR_AGGR_', prefix, sec_label_str, '.png']), '-dpng', '-r150');
        print(local_fig_client_in, fullfile(figures_folder, ['IR_CLIENT_IN_', prefix, sec_label_str, '.png']), '-dpng', '-r150');
        print(local_fig_client_out, fullfile(figures_folder, ['IR_CLIENT_OUT_', prefix, sec_label_str, '.png']), '-dpng', '-r150');
        
        fprintf('Figures saved to: %s\n', figures_folder);
    end
    
    disp(' *** FINISHED CREATING THE IRs ***');
    fprintf('It took %.4f seconds to generate and print the graphs.\n', toc);
end

%% ==================== Styling Functions ====================

function style = get_plot_style()
    % Define consistent styling for all plots
    style = struct();
    
    % Colors (professional palette)
    style.colors = struct();
    style.colors.primary = [0.00, 0.45, 0.74];    % Blue
    style.colors.secondary = [0.85, 0.33, 0.10];  % Orange/Red
    style.colors.tertiary = [0.47, 0.67, 0.19];   % Green
    style.colors.baseline = [0.50, 0.50, 0.50];   % Gray
    style.colors.grid = [0.85, 0.85, 0.85];       % Light gray
    
    % Line widths
    style.linewidth_main = 2.0;
    style.linewidth_baseline = 1.2;
    
    % Font sizes
    style.title_fontsize = 16;
    style.subtitle_fontsize = 13;
    style.axis_fontsize = 11;
    style.legend_fontsize = 10;
    
    % Grid
    style.show_grid = true;
    style.grid_alpha = 0.5;
end

function labels = get_default_legend_labels(n_series)
    if n_series == 1
        labels = {'Perfect Foresight'};
    elseif n_series == 2
        labels = {'Perfect Foresight', 'First-Order'};
    else
        labels = {'Perfect Foresight', 'First-Order', 'Second-Order'};
    end
end

function specs = get_line_specs(n_series, style)
    % Return line specifications: {color, linewidth, linestyle}
    if n_series == 1
        specs = {{style.colors.primary, style.linewidth_main, '-'}};
    elseif n_series == 2
        specs = {{style.colors.primary, style.linewidth_main, '-'}, ...
                 {style.colors.secondary, style.linewidth_main, '--'}};
    else
        specs = {{style.colors.primary, style.linewidth_main, '-'}, ...
                 {style.colors.secondary, style.linewidth_main, '--'}, ...
                 {style.colors.tertiary, style.linewidth_main, '-.'}};
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

function val = get_field_or_default(s, field, default)
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

function str_out = escape_latex(str_in)
    % Escape special LaTeX characters for safe use in titles
    str_out = str_in;
    str_out = strrep(str_out, '\', '\\');
    str_out = strrep(str_out, '_', '\_');
    str_out = strrep(str_out, '%', '\%');
    str_out = strrep(str_out, '&', '\&');
    str_out = strrep(str_out, '#', '\#');
    str_out = strrep(str_out, '$', '\$');
    str_out = strrep(str_out, '{', '\{');
    str_out = strrep(str_out, '}', '\}');
    str_out = strrep(str_out, '^', '\^{}');
    str_out = strrep(str_out, '~', '\~{}');
end

%% ==================== Core Plotting Function ====================

function plot_subplot_multi(ax, baseline, d1, d2, d3, row, N, range_padding, specs, t_off, style)
    hold on;
    
    % Plot baseline (steady state)
    h_baseline = plot(ax, baseline * ones(N, 1), 'Color', style.colors.baseline, ...
        'LineWidth', style.linewidth_baseline, 'LineStyle', ':');
    
    t1 = 1 + t_off;
    t2 = N + t_off;
    
    % Plot main series
    h1 = plot(ax, d1(row, t1:t2), 'Color', specs{1}{1}, ...
        'LineWidth', specs{1}{2}, 'LineStyle', specs{1}{3});
    
    if ~isempty(d2)
        h2 = plot(ax, d2(row, t1:t2), 'Color', specs{2}{1}, ...
            'LineWidth', specs{2}{2}, 'LineStyle', specs{2}{3});
    end
    if ~isempty(d3)
        h3 = plot(ax, d3(row, t1:t2), 'Color', specs{3}{1}, ...
            'LineWidth', specs{3}{2}, 'LineStyle', specs{3}{3});
    end
    hold off;
    
    % Y-axis formatting
    YLim = get(gca, 'YLim');
    y_range = range(YLim);
    if y_range > 0
        padding = range_padding * y_range;
        set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]);
    end
    
    % Styling
    set(gca, 'FontSize', style.axis_fontsize);
    set(gca, 'Box', 'on');
    set(gca, 'TickDir', 'out');
    
    % Grid
    if style.show_grid
        grid on;
        set(gca, 'GridColor', style.colors.grid, 'GridAlpha', style.grid_alpha);
    end
    
    % X-axis label
    xlabel('Periods', 'FontSize', style.axis_fontsize);
    
    % Remove exponent notation for y-axis
    yaxis = get(gca, 'YAxis');
    yaxis.Exponent = 0;
end

%% ==================== Panel Plotting Functions ====================

function plot_aggregate_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off, style)
    n_cols = 2;
    n_rows = 2;
    
    subplot(n_rows, n_cols, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0, style);
    title('$A_j$ (TFP)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
    set(leg, 'Box', 'off');
    
    subplot(n_rows, n_cols, 2);
    plot_subplot_multi(ax, 0, d1, d2, d3, 2, N, padding, specs, t_off, style);
    title('$C$ (Consumption)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    
    subplot(n_rows, n_cols, 3);
    plot_subplot_multi(ax, 0, d1, d2, d3, 3, N, padding, specs, t_off, style);
    title('$L$ (Labor)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    
    subplot(n_rows, n_cols, 4);
    plot_subplot_multi(ax, 0, d1, d2, d3, 24, N, padding, specs, t_off, style);
    title('$Y$ (Output)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
end

function plot_sector_inputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off, style)
    subplot(2, 3, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0, style);
    title('$A_j$ (TFP)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
    set(leg, 'Box', 'off');
    
    subplot(2, 3, 2);
    plot_subplot_multi(ax, 0, d1, d2, d3, 8, N, padding, specs, t_off, style);
    title('$L_{j}$ (Labor)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    
    subplot(2, 3, 3);
    plot_subplot_multi(ax, 0, d1, d2, d3, 9, N, padding, specs, t_off, style);
    title('$I_{j}$ (Investment)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    
    subplot(2, 3, 4);
    plot_subplot_multi(ax, 0, d1, d2, d3, 10, N, padding, specs, t_off, style);
    title('$M_{j}$ (Intermediates)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    
    subplot(2, 3, 5);
    plot_subplot_multi(ax, 0, d1, d2, d3, 11, N, padding, specs, t_off, style);
    title('$Y_{j}$ (Value Added)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
end

function plot_sector_outputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off, style)
    subplot(2, 3, 1);
    plot_subplot_multi(ax, 1, d1, d2, d3, 1, N, padding, specs, 0, style);
    title('$A_j$ (TFP)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
    set(leg, 'Box', 'off');
    
    subplot(2, 3, 2);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 5, N, padding, specs, t_off, style);
        title('$P_{j}$ (Price)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 4, N, padding, specs, t_off, style);
        title('$C_{j}$ (Consumption)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
    
    subplot(2, 3, 3);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 12, N, padding, specs, t_off, style);
        title('$Q_{j}$ (Gross Output)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 5, N, padding, specs, t_off, style);
        title('$P_{j}$ (Price)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
    
    subplot(2, 3, 4);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 4, N, padding, specs, t_off, style);
        title('$C_{j}$ (Consumption)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 7, N, padding, specs, t_off, style);
        title('$M^{out}_j$ (Intermediate Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
    
    subplot(2, 3, 5);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 7, N, padding, specs, t_off, style);
        title('$M^{out}_j$ (Intermediate Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 6, N, padding, specs, t_off, style);
        title('$I^{out}_j$ (Investment Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
    
    subplot(2, 3, 6);
    if n_series == 3
        plot_subplot_multi(ax, 0, d1, d2, d3, 6, N, padding, specs, t_off, style);
        title('$I^{out}_j$ (Investment Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        plot_subplot_multi(ax, 0, d1, d2, d3, 12, N, padding, specs, t_off, style);
        title('$Q_{j}$ (Gross Output)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
end

function plot_client_inputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off, style)
    if n_series < 3
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 1, d1, d2, d3, 13, N, padding, specs, 0, style);
        title('$A_j$ (TFP)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
        set(leg, 'Box', 'off');
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 18, N, padding, specs, t_off, style);
        title('$L_j$ (Labor)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 19, N, padding, specs, t_off, style);
        title('$I_j$ (Investment)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 20, N, padding, specs, t_off, style);
        title('$M_j$ (Intermediates)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 21, N, padding, specs, t_off, style);
        title('$Y_j$ (Value Added)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 0, d1, d2, d3, 18, N, padding, specs, t_off, style);
        title('$L_j$ (Labor)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 19, N, padding, specs, t_off, style);
        title('$I_j$ (Investment)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 20, N, padding, specs, t_off, style);
        title('$M_j$ (Intermediates)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
        set(leg, 'Box', 'off');
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 21, N, padding, specs, t_off, style);
        title('$Y_j$ (Value Added)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 25, N, padding, specs, t_off, style);
        title('$P^m_{j}$ (Intermediate Price)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 6);
        plot_subplot_multi(ax, 0, d1, d2, d3, 26, N, padding, specs, t_off, style);
        title('$\hat{\gamma}_{ij}$ (Expend. Share)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
end

function plot_client_outputs_panel(ax, N, d1, d2, d3, n_series, padding, specs, legend_entries, t_off, style)
    if n_series < 3
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 1, d1, d2, d3, 13, N, padding, specs, 0, style);
        title('$A_j$ (TFP)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
        set(leg, 'Box', 'off');
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 14, N, padding, specs, t_off, style);
        title('$C_j$ (Consumption)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 15, N, padding, specs, t_off, style);
        title('$P_j$ (Price)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 17, N, padding, specs, t_off, style);
        title('$M^{out}_j$ (Intermediate Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 16, N, padding, specs, t_off, style);
        title('$I^{out}_j$ (Investment Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 6);
        plot_subplot_multi(ax, 0, d1, d2, d3, 22, N, padding, specs, t_off, style);
        title('$Q_j$ (Gross Output)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    else
        subplot(2, 3, 1);
        plot_subplot_multi(ax, 0, d1, d2, d3, 15, N, padding, specs, t_off, style);
        title('$P_j$ (Price)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 2);
        plot_subplot_multi(ax, 0, d1, d2, d3, 22, N, padding, specs, t_off, style);
        title('$Q_j$ (Gross Output)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 3);
        plot_subplot_multi(ax, 0, d1, d2, d3, 14, N, padding, specs, t_off, style);
        title('$C_j$ (Consumption)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        leg = legend(legend_entries, 'Location', 'best', 'FontSize', style.legend_fontsize);
        set(leg, 'Box', 'off');
        
        subplot(2, 3, 4);
        plot_subplot_multi(ax, 0, d1, d2, d3, 17, N, padding, specs, t_off, style);
        title('$M^{out}_j$ (Intermediate Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
        
        subplot(2, 3, 5);
        plot_subplot_multi(ax, 0, d1, d2, d3, 16, N, padding, specs, t_off, style);
        title('$I^{out}_j$ (Investment Sales)', 'Interpreter', 'latex', 'FontSize', style.subtitle_fontsize);
    end
end
