function GraphIRs(irs_1,irs_2, irs_3, ax, N, labels, range_padding, save_label, save_exper_ind)
% function graphs = Graphs_LogLinearIRS(irs_1, ax,N,labels,range_padding,save_exper_ind)
    % Local figures
    local_fig_agg = figure('Position', [100, 100, 800, 800]);               
    local_fig_sec_in = figure('Position', [100, 100, 800, 800]);            
    local_fig_sec_out = figure('Position', [100, 100, 800, 800]);
    local_fig_client_in = figure('Position', [100, 100, 800, 800]);
    local_fig_client_out = figure('Position', [100, 100, 800, 800]);
tic;
   for idx = 1:numel(labels{1,1})
        sector_idx = labels{1,1}(idx);
        sector_label = labels{1,2}{idx};
        client_idx = labels{1,3}(idx);
        client_label = labels{1,4}{idx};
        title_sector = strcat(sector_label, ' shock');
    
        if isempty(irs_2) && isempty(irs_3)       
            % IN
            figure(local_fig_sec_in);
        
            subplot(2, 3, 1);
            plot(ax, 1 * ones(N, 1), '--k', ax, irs_1{idx}(1, 1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  % Calculate padding (5% of the range)
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); % Set new y-axis limits
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1)},'Location','southeast');
        
            subplot(2, 3, 2);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(9,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 3);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(10,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 4);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(11,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 5);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(12,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % OUT
            figure(local_fig_sec_out);
        
            subplot(2, 3, 1);
            plot(ax, 1 * ones(N, 1), '--k', ax, irs_1{idx}(1, 1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1)},'Location','southeast');
        
            subplot(2, 3, 2);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(5,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 3);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(6,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 4);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(8,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 5);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(7,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2, 3, 6);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(13,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]);
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_j$','Interpreter', 'latex',  'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % AGGR
            figure(local_fig_agg);
        
            subplot(2, 3, 1);
            plot(ax, 1 * ones(N, 1), '--k', ax, irs_1{idx}(1, 1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1)},'Location','southeast');
        
            subplot(2, 3, 2);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(2,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 3);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(3,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 4);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(25,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 5);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(4,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Vc$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{Aggregate}', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
        
            % IN (CLIENTS)    
            figure(local_fig_client_in);
        
            subplot(2, 3, 1);
            plot(ax, 1 * ones(N, 1), '--k', ax, irs_1{idx}(14,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  % Calculate padding (5% of the range)
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); % Set new y-axis limits
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1)},'Location','southeast');
        
            subplot(2, 3, 2);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(19,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 3);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(20,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 4);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(21,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 5);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(22,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, client_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
        
            % OUT (CLIENTS)
            figure(local_fig_client_out);
        
            subplot(2, 3, 1);
            plot(ax, 1 * ones(N, 1), '--k', ax, irs_1{idx}(14,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1)},'Location','southeast');
        
            subplot(2, 3, 2);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(15,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 3);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(16,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 4);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(18,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2, 3, 5);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(17,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2, 3, 6);
            plot(ax, 0 * ones(N, 1), '--k', ax, irs_1{idx}(23,1:N), 'k', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]);
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, client_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);

        elseif isempty(irs_3)

            % IN
            figure(local_fig_sec_in);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  % Calculate padding (5% of the range)
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); % Set new y-axis limits
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1), inputname(2)}, 'Location','southeast');
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(9,1:N),'k', ax,irs_2{idx}(9,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(10,1:N),'k', ax,irs_2{idx}(10,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(11,1:N),'k', ax,irs_2{idx}(11,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(12,1:N),'k', ax,irs_2{idx}(12,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % OUT
            figure(local_fig_sec_out);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1), inputname(2)}, 'Location','southeast');
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(5,1:N),'k', ax,irs_2{idx}(5,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(6,1:N),'k', ax,irs_2{idx}(6,1:N),'--r','LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(8,1:N),'k', ax,irs_2{idx}(8,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(7,1:N),'k', ax,irs_2{idx}(7,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,6);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(13,1:N),'k', ax,irs_2{idx}(13,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
            
            % AGGR
            figure(local_fig_agg);
            
            subplot(2,2,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1), inputname(2)}, 'Location','southeast');
            
            subplot(2,2,2);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(2,1:N),'k', ax,irs_2{idx}(2,1:N),'--r','LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,2,3);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(3,1:N),'k', ax,irs_2{idx}(3,1:N),'--r','LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,2,4);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(4,1:N),'k', ax,irs_2{idx}(4,1:N),'--r','LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Vc$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);    
            
            title_ = sprintf('%s, effect on \\textbf{Aggregate}', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
        
            % IN (CLIENT)
            figure(local_fig_client_in);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(14,1:N),'k', ax,irs_2{idx}(14,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1), inputname(2)}, 'Location','southeast');
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(19,1:N),'k', ax,irs_2{idx}(19,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(20,1:N),'k', ax,irs_2{idx}(20,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(21,1:N),'k', ax,irs_2{idx}(21,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(22,1:N),'k', ax,irs_2{idx}(22,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % OUT (CLIENT)
            figure(local_fig_client_out);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(14,1:N),'k', ax,irs_2{idx}(14,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', inputname(1), inputname(2)}, 'Location','southeast');
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(15,1:N),'k', ax,irs_2{idx}(15,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(16,1:N),'k', ax,irs_2{idx}(16,1:N),'--r','LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(18,1:N),'k', ax,irs_2{idx}(18,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(17,1:N),'k', ax,irs_2{idx}(17,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,6);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(23,1:N),'k', ax,irs_2{idx}(23,1:N),'--r','LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, client_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);

        elseif ~isempty(irs_1) && ~isempty(irs_2) && ~isempty(irs_3)
            
            % AGGR
            figure(local_fig_agg);
            
            subplot(2,2,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r',ax,irs_3{idx}(1,1:N), '-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', 'Nonlinear CES','Loglinear CES','Cobb-Douglas'},'Location','southeast');
            
            subplot(2,2,2);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(2,2:N+1),'k', ax,irs_2{idx}(2,2:N+1),'--r', ax,irs_3{idx}(2,2:N+1),'-.b', 'LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,2,3);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(3,2:N+1),'k', ax,irs_2{idx}(3,2:N+1),'--r', ax,irs_3{idx}(3,2:N+1),'-.b', 'LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,2,4);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(4,2:N+1),'k', ax,irs_2{idx}(4,2:N+1),'--r', ax,irs_3{idx}(4,2:N+1),'-.b', 'LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Vc$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);    
            
            title_ = sprintf('%s, effect on \\textbf{Aggregate}', title_sector);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);

            % IN
            figure(local_fig_sec_in);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r',ax, irs_3{idx}(1,1:N), '-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  % Calculate padding (5% of the range)
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); % Set new y-axis limits
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', 'Nonlinear CES','Loglinear CES','Cobb-Douglas'},'Location','southeast');
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(9,2:N+1),'k', ax,irs_2{idx}(9,2:N+1),'--r', ax,irs_3{idx}(9,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(10,2:N+1),'k', ax,irs_2{idx}(10,2:N+1),'--r', ax,irs_3{idx}(10,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(11,2:N+1),'k', ax,irs_2{idx}(11,2:N+1),'--r', ax,irs_3{idx}(11,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(12,2:N+1),'k', ax,irs_2{idx}(12,2:N+1),'--r', ax,irs_3{idx}(12,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % OUT
            figure(local_fig_sec_out);
            
            subplot(2,3,1);
            plot(ax, 1 * ones(N, 1), '--k', ax,irs_1{idx}(1,1:N),'k', ax,irs_2{idx}(1,1:N),'--r',ax, irs_3{idx}(1,1:N), '-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$A_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', 'Nonlinear CES','Loglinear CES','Cobb-Douglas'},'Location','southeast');

            subplot(2,3,2);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(6,2:N+1),'k', ax,irs_2{idx}(6,2:N+1),'--r', ax,irs_3{idx}(6,2:N+1),'-.b', 'LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(13,2:N+1),'k', ax,irs_2{idx}(13,2:N+1),'--r', ax,irs_3{idx}(13,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(5,2:N+1),'k', ax,irs_2{idx}(5,2:N+1),'--r', ax,irs_3{idx}(5,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(8,2:N+1),'k', ax,irs_2{idx}(8,2:N+1),'--r', ax,irs_3{idx}(8,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,6);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(7,2:N+1),'k', ax,irs_2{idx}(7,2:N+1),'--r', ax,irs_3{idx}(7,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, sector_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
            

            % IN (CLIENT)
            figure(local_fig_client_in);
            
            subplot(2,3,1);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(19,2:N+1),'k', ax,irs_2{idx}(19,2:N+1),'--r', ax,irs_3{idx}(19,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$L_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            
            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(20,2:N+1),'k', ax,irs_2{idx}(20,2:N+1),'--r', ax,irs_3{idx}(20,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1), '--k', ax,irs_1{idx}(21,2:N+1),'k', ax,irs_2{idx}(21,2:N+1),'--r', ax,irs_3{idx}(21,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', 'Nonlinear CES','Loglinear CES','Cobb-Douglas'},'Location','southeast');
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(22,2:N+1),'k', ax,irs_2{idx}(22,2:N+1),'--r', ax,irs_3{idx}(22,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Y_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(26,2:N+1),'k', ax,irs_2{idx}(26,2:N+1),'--r', ax,irs_3{idx}(26,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P^m_{j}$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,6);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(27,2:N+1),'k', ax,irs_2{idx}(27,2:N+1),'--r', ax,irs_3{idx}(27,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$\hat{\gamma}_{ij}$ (Expend. Share)', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            
            
            title_ = sprintf('%s, effect on \\textbf{%s} (INPUTS)', title_sector, client_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        
            % OUT (CLIENT)
            figure(local_fig_client_out);     
            
            subplot(2,3,1);
            plot(ax,0*ones(N,1),'--k', ax,irs_1{idx}(16,2:N+1),'k', ax,irs_2{idx}(16,2:N+1),'--r', ax,irs_3{idx}(16,2:N+1),'-.b', 'LineWidth',1.5)
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$P_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            

            subplot(2,3,2);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(23,2:N+1),'k', ax,irs_2{idx}(23,2:N+1),'--r', ax,irs_3{idx}(23,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$Q_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);

            subplot(2,3,3);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(15,2:N+1),'k', ax,irs_2{idx}(15,2:N+1),'--r', ax,irs_3{idx}(15,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$C_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
            legend({'Steady State', 'Nonlinear CES','Loglinear CES','Cobb-Douglas'},'Location','southeast');
            
            subplot(2,3,4);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(18,2:N+1),'k', ax,irs_2{idx}(18,2:N+1),'--r', ax,irs_3{idx}(18,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$M^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
        
            subplot(2,3,5);
            plot(ax, 0 * ones(N, 1),'--k', ax,irs_1{idx}(17,2:N+1),'k', ax,irs_2{idx}(17,2:N+1),'--r', ax,irs_3{idx}(17,2:N+1),'-.b', 'LineWidth', 1.5);
            YLim = get(gca, 'YLim');   
            padding = range_padding * range(YLim);  
            set(gca, 'YLim', [YLim(1) - padding, YLim(2) + padding]); 
            yaxis = get(gca, 'YAxis');
            yaxis.Exponent = 0;
            title('$I^{out}_j$', 'Interpreter', 'latex', 'Fontsize', 12);
            set(gca, 'Fontsize', 12);
       
            title_ = sprintf('%s, effect on \\textbf{%s} (OUTPUTS)', title_sector, client_label);
            sgtitle(title_, 'Interpreter', 'latex', 'Fontsize', 18);
        else
        % Incorrect input case
            fprintf('Incorrect input for the function. Please read the documentation for the function.\n');
        end
    end

    if save_exper_ind==1
        filename_sec_in = strcat('output/IR_IN', save_label, sector_label, '.png');
        saveas(local_fig_sec_in,filename_sec_in)
        filename_sec_out = strcat('output/IR_OUT', save_label, sector_label , '.png');
        saveas(local_fig_sec_out,filename_sec_out)
        filename_agg = strcat('output/IR_AGGR', save_label, sector_label , '.png');
        saveas(local_fig_agg,filename_agg)
        filename_sec_in = strcat('output/IR_CLIENT_IN', save_label, sector_label, '.png');
        saveas(local_fig_client_in,filename_sec_in)
        filename_sec_out = strcat('output/IR_CLIENT_OUT', save_label, sector_label , '.png');
        saveas(local_fig_client_out,filename_sec_out)
    end
        disp(' *** FINISHED CREATING THE IRs ***')
        elapsed_time = toc;
        fprintf('It took %.4f seconds to generate and print the graphs.\n', elapsed_time);
        disp('The 5 IR figures have been saved in the folder "output"')
end
