function plot_ir_subplot(ax, baseline, data, N, title_str, range_padding, line_specs, legend_entries)
% PLOT_IR_SUBPLOT Plot a single impulse response subplot
%
% INPUTS:
%   ax            - x-axis values (0:N-1)
%   baseline      - Baseline value (0 or 1 typically)
%   data          - Cell array of data series to plot {data1, data2, ...}
%   N             - Number of periods to plot
%   title_str     - Title for the subplot (LaTeX format)
%   range_padding - Padding for y-axis range (fraction)
%   line_specs    - Cell array of line specifications {{'k', 1.5}, {'--r', 1.5}, ...}
%   legend_entries- Cell array of legend entries (optional, only first subplot)

    hold on;
    
    % Plot baseline
    plot(ax, baseline * ones(N, 1), '--k', 'LineWidth', 1);
    
    % Plot each data series
    for i = 1:numel(data)
        if ~isempty(data{i})
            plot(ax, data{i}(1:N), line_specs{i}{1}, 'LineWidth', line_specs{i}{2});
        end
    end
    
    hold off;
    
    % Adjust y-axis limits
    YLim = get(gca, 'YLim');
    padding = range_padding * range(YLim);
    set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]);
    
    % Format axis
    yaxis = get(gca, 'YAxis');
    yaxis.Exponent = 0;
    
    % Set title
    title(title_str, 'Interpreter', 'latex', 'Fontsize', 12);
    set(gca, 'Fontsize', 12);
    
    % Add legend if provided
    if nargin >= 8 && ~isempty(legend_entries)
        legend(legend_entries, 'Location', 'southeast');
    end
end

