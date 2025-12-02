% This code compares the descriptive statistics of the TFP process
% parameters (the persistence parameter, rho
% and the elements of the variance-covariance matrix) for different ways of
% computing them.

clear all; clc; 
load TFP_process_vLW.mat;
modrho_vLW = modrho;
modvcv_vLW = modvcv;
clear modrho modvcv;
load TFP_process.mat;

% Descriptive Statistics for rho
comp_modrho = [modrho, modrho_sm, modrho_GO, modrho_GO_sm, modrho_GO_noVA, modrho_GO_noVA_sm, modrho_vLW];
Statistics_modrho = [mean(comp_modrho); std(comp_modrho); var(comp_modrho); min(comp_modrho); prctile(comp_modrho,25); median(comp_modrho); prctile(comp_modrho,75) ; max(comp_modrho)];
rowNames = {'Mean','St. Deviation','Variance','Min','25th Prctle','Median','75th Prctle','Max'};
variableNames_rho = {'modrho', 'modrho_sm', 'modrho_GO', 'modrho_GO_sm', 'modrho_GO_noVA', 'modrho_GO_noVA_sm', 'modrho_vLW'};
Statistics_modrho = array2table(Statistics_modrho,'VariableNames',variableNames_rho, 'RowNames',rowNames);

% Descriptive Statistics for Main Diagnonal of Variance-Covariance matrix
comp_vcv = [diag(modvcv), diag(modvcv_sm), diag(modvcv_GO), diag(modvcv_GO_sm), diag(modvcv_GO_noVA), diag(modvcv_GO_noVA_sm), diag(modvcv_vLW)];
Statistics_Diagonal_vcv = [mean(comp_vcv); std(comp_vcv); var(comp_vcv); min(comp_vcv); prctile(comp_vcv,25); median(comp_vcv); prctile(comp_vcv,75) ; max(comp_vcv)];
variableNames_vcv = {'vcv', 'vcv_sm', 'vcv_GO', 'vcv_GO_sm', 'vcv_GO_noVA', 'vcv_GO_noVA_sm', 'vcv_vLW'};
Statistics_Diagonal_vcv = array2table(Statistics_Diagonal_vcv,'VariableNames',variableNames_vcv, 'RowNames',rowNames);

% Descriptive Statistics for Off-Diagonal elements of Variance-Covariance matrix
Statistics_OffDiagonal_vcv = cell(1,36);
for i = 1:36
    if i == 36
        Statistics_OffDiagonal_vcv{1,i} = [diag(modvcv,-i), diag(modvcv_sm,-i), diag(modvcv_GO_noVA,-i), diag(modvcv_GO_noVA_sm,-i), diag(modvcv_vLW,-i)];
        continue;
    end
        comp_vcv = [diag(modvcv,-i), diag(modvcv_sm,-i), diag(modvcv_GO,-i), diag(modvcv_GO_sm,-i), diag(modvcv_GO_noVA,-i), diag(modvcv_GO_noVA_sm,-i), diag(modvcv_vLW,-i)];
        stats = [mean(comp_vcv); std(comp_vcv); var(comp_vcv); min(comp_vcv); prctile(comp_vcv,25); median(comp_vcv); prctile(comp_vcv,75) ; max(comp_vcv)];
        Statistics_OffDiagonal_vcv{1,i} = array2table(stats,'VariableNames',variableNames_vcv, 'RowNames',rowNames);
end

% Descriptive Statistics for Sectoral Shock Standard Deviations
% std(ar1resid) gives the shock volatility for each sector
shock_std = [std(ar1resid)', std(ar1resid_sm)', std(ar1resid_GO)', std(ar1resid_GO_sm)', std(ar1resid_GO_noVA)', std(ar1resid_GO_noVA_sm)'];
Statistics_shock_std = [mean(shock_std); std(shock_std); var(shock_std); min(shock_std); prctile(shock_std,25); median(shock_std); prctile(shock_std,75); max(shock_std)];
variableNames_shock = {'VA', 'VA_sm', 'GO', 'GO_sm', 'GO_noVA', 'GO_noVA_sm'};
Statistics_shock_std = array2table(Statistics_shock_std,'VariableNames',variableNames_shock, 'RowNames',rowNames);

% Full sectoral shock std (each row = sector, each column = method)
Sectoral_shock_std = array2table(shock_std, 'VariableNames', variableNames_shock);

% Correlation of shocks across sectors (average pairwise correlation)
corr_shocks = [mean(nonzeros(triu(corr(ar1resid),1))), ...
               mean(nonzeros(triu(corr(ar1resid_sm),1))), ...
               mean(nonzeros(triu(corr(ar1resid_GO),1))), ...
               mean(nonzeros(triu(corr(ar1resid_GO_sm),1))), ...
               mean(nonzeros(triu(corr(ar1resid_GO_noVA),1))), ...
               mean(nonzeros(triu(corr(ar1resid_GO_noVA_sm),1)))];
Avg_shock_correlation = array2table(corr_shocks, 'VariableNames', variableNames_shock);

clearvars -except Statistics_modrho Statistics_Diagonal_vcv Statistics_OffDiagonal_vcv Statistics_shock_std Sectoral_shock_std Avg_shock_correlation;

