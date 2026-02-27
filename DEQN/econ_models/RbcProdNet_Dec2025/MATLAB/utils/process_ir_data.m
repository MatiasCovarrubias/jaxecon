function [irs, sectoral] = process_ir_data(dynare_simul, sector_idx, client_idx, params, steady_state, ...
    n_sectors, endostates_ss, Cagg_ss, Lagg_ss, policies_ss)
% PROCESS_IR_DATA Process Dynare simulation output into impulse response data
%
% This function extracts and transforms variables from Dynare simulation output
% into deviations from steady state for plotting impulse responses.
%
% INPUTS:
%   dynare_simul  - Dynare simulation output matrix (n_vars x T)
%   sector_idx    - Index of the shocked sector
%   client_idx    - Index of the main client sector
%   params        - Model parameters (needs Gamma_M, sigma_m)
%   steady_state  - Dynare steady state vector
%   n_sectors     - Number of sectors (37)
%   endostates_ss - Steady state endogenous states (log capital)
%   Cagg_ss       - Aggregate consumption steady state
%   Lagg_ss       - Aggregate labor steady state
%   policies_ss   - Steady state policy variables
%
% OUTPUTS:
%   irs - Matrix of impulse responses (27 x T):
%         Row  1: A_ir (TFP level)
%         Row  2: C_ir (aggregate consumption expenditure, log dev from SS)
%         Row  3: I_ir (aggregate investment expenditure, log dev from SS)
%         Row  4: Cj_ir (sectoral consumption)
%         Row  5: Pj_ir (sectoral price)
%         Row  6: Ioutj_ir (sectoral investment output)
%         Row  7: Moutj_ir (sectoral intermediate output)
%         Row  8: Lj_ir (sectoral labor)
%         Row  9: Ij_ir (sectoral investment input)
%         Row 10: Mj_ir (sectoral intermediate input)
%         Row 11: Yj_ir (sectoral output)
%         Row 12: Qj_ir (sectoral Tobin's Q)
%         Row 13: A_client_ir (client TFP level)
%         Row 14: Cj_client_ir (client consumption)
%         Row 15: Pj_client_ir (client price)
%         Row 16: Ioutj_client_ir (client investment output)
%         Row 17: Moutj_client_ir (client intermediate output)
%         Row 18: Lj_client_ir (client labor)
%         Row 19: Ij_client_ir (client investment input)
%         Row 20: Mj_client_ir (client intermediate input)
%         Row 21: Yj_client_ir (client output)
%         Row 22: Qj_client_ir (client Tobin's Q)
%         Row 23: Kj_ir (sectoral capital)
%         Row 24: GDP_ir (aggregate GDP expenditure, log dev from SS)
%         Row 25: Pmj_client_ir (client intermediate price)
%         Row 26: gammaij_client_ir (client expenditure share deviation)
%         Row 27: C_util_ir (utility aggregate consumption, log dev from SS)

    %% Input validation
    validate_params(params, {'Gamma_M', 'sigma_m'}, 'process_ir_data');
    
    %% Get standardized indices
    idx = get_variable_indices(n_sectors);
    
    %% Helper functions to get specific sector index within a range
    % Dynare indices include states (k, a), so c starts at 2*n+1
    % policies_ss indices start at 1 (no states), so use ss_offset to convert
    dyn_idx = @(range, j) range(1) + j - 1;  % Dynare variable index for sector j
    pol_idx = @(dyn_range, j) dyn_range(1) - idx.ss_offset + j - 1;  % policies_ss index for sector j
    
    %% Aggregate variables
    A_ir = exp(dynare_simul(dyn_idx(idx.a, sector_idx), :));
    % Expenditure-based aggregates (nominal, then log dev from SS)
    epsilon = 1e-12;
    p_log = dynare_simul(idx.p(1):idx.p(2), :);
    c_log = dynare_simul(idx.c(1):idx.c(2), :);
    iout_log = dynare_simul(idx.iout(1):idx.iout(2), :);
    q_log = dynare_simul(idx.q(1):idx.q(2), :);
    mout_log = dynare_simul(idx.mout(1):idx.mout(2), :);
    
    p_ss_log = policies_ss((idx.p(1)-idx.ss_offset):(idx.p(2)-idx.ss_offset));
    c_ss_log = policies_ss((idx.c(1)-idx.ss_offset):(idx.c(2)-idx.ss_offset));
    iout_ss_log = policies_ss((idx.iout(1)-idx.ss_offset):(idx.iout(2)-idx.ss_offset));
    q_ss_log = policies_ss((idx.q(1)-idx.ss_offset):(idx.q(2)-idx.ss_offset));
    mout_ss_log = policies_ss((idx.mout(1)-idx.ss_offset):(idx.mout(2)-idx.ss_offset));
    
    C_nom_levels = sum(exp(p_log + c_log), 1);
    I_nom_levels = sum(exp(p_log + iout_log), 1);
    GDP_nom_levels = sum(exp(p_log + q_log) - exp(p_log + mout_log), 1);
    
    C_nom_ss = sum(exp(p_ss_log + c_ss_log));
    I_nom_ss = sum(exp(p_ss_log + iout_ss_log));
    GDP_nom_ss = sum(exp(p_ss_log + q_ss_log) - exp(p_ss_log + mout_ss_log));
    
    C_ir = log(max(C_nom_levels, epsilon)) - log(max(C_nom_ss, epsilon));
    I_ir = log(max(I_nom_levels, epsilon)) - log(max(I_nom_ss, epsilon));
    GDP_ir = log(max(GDP_nom_levels, epsilon)) - log(max(GDP_nom_ss, epsilon));
    C_util_ir = dynare_simul(idx.cagg, :) - log(Cagg_ss);
    
    %% Sectoral output variables (shocked sector)
    Cj_ir = dynare_simul(dyn_idx(idx.c, sector_idx), :) - policies_ss(pol_idx(idx.c, sector_idx));
    Pj_ir = dynare_simul(dyn_idx(idx.p, sector_idx), :) - policies_ss(pol_idx(idx.p, sector_idx));
    Ioutj_ir = dynare_simul(dyn_idx(idx.iout, sector_idx), :) - policies_ss(pol_idx(idx.iout, sector_idx));
    Moutj_ir = dynare_simul(dyn_idx(idx.mout, sector_idx), :) - policies_ss(pol_idx(idx.mout, sector_idx));
    
    %% Sectoral input variables (shocked sector)
    Lj_ir = dynare_simul(dyn_idx(idx.l, sector_idx), :) - policies_ss(pol_idx(idx.l, sector_idx));
    Ij_ir = dynare_simul(dyn_idx(idx.i, sector_idx), :) - policies_ss(pol_idx(idx.i, sector_idx));
    Mj_ir = dynare_simul(dyn_idx(idx.m, sector_idx), :) - policies_ss(pol_idx(idx.m, sector_idx));
    Yj_ir = dynare_simul(dyn_idx(idx.y, sector_idx), :) - policies_ss(pol_idx(idx.y, sector_idx));
    Qj_ir = dynare_simul(dyn_idx(idx.q, sector_idx), :) - policies_ss(pol_idx(idx.q, sector_idx));
    Kj_ir = dynare_simul(dyn_idx(idx.k, sector_idx), :) - endostates_ss(sector_idx);
    
    %% Client sectoral output variables
    A_client_ir = exp(dynare_simul(dyn_idx(idx.a, client_idx), :));
    Cj_client_ir = dynare_simul(dyn_idx(idx.c, client_idx), :) - policies_ss(pol_idx(idx.c, client_idx));
    Pj_client_ir = dynare_simul(dyn_idx(idx.p, client_idx), :) - policies_ss(pol_idx(idx.p, client_idx));
    Ioutj_client_ir = dynare_simul(dyn_idx(idx.iout, client_idx), :) - policies_ss(pol_idx(idx.iout, client_idx));
    Moutj_client_ir = dynare_simul(dyn_idx(idx.mout, client_idx), :) - policies_ss(pol_idx(idx.mout, client_idx));
    
    %% Client sectoral input variables
    Lj_client_ir = dynare_simul(dyn_idx(idx.l, client_idx), :) - policies_ss(pol_idx(idx.l, client_idx));
    Pmj_client_ir = dynare_simul(dyn_idx(idx.pm, client_idx), :) - policies_ss(pol_idx(idx.pm, client_idx));
    Mj_client_ir = dynare_simul(dyn_idx(idx.m, client_idx), :) - policies_ss(pol_idx(idx.m, client_idx));
    Ij_client_ir = dynare_simul(dyn_idx(idx.i, client_idx), :) - policies_ss(pol_idx(idx.i, client_idx));
    Yj_client_ir = dynare_simul(dyn_idx(idx.y, client_idx), :) - policies_ss(pol_idx(idx.y, client_idx));
    Qj_client_ir = dynare_simul(dyn_idx(idx.q, client_idx), :) - policies_ss(pol_idx(idx.q, client_idx));
    
    %% Client expenditure share on affected input
    Pmj_lev_client_ir = exp(dynare_simul(dyn_idx(idx.pm, client_idx), :));
    Pmj_lev_client_ss = exp(policies_ss(pol_idx(idx.pm, client_idx)));
    Pj_lev_ir = exp(dynare_simul(dyn_idx(idx.p, sector_idx), :));
    Pj_lev_ss = exp(policies_ss(pol_idx(idx.p, sector_idx)));
    
    gammaij_lev_client_ir = params.Gamma_M(sector_idx, client_idx) .* ...
        (Pj_lev_ir ./ Pmj_lev_client_ir).^(1 - params.sigma_m);
    gammaij_lev_client_ss = params.Gamma_M(sector_idx, client_idx) * ...
        (Pj_lev_ss / Pmj_lev_client_ss)^(1 - params.sigma_m);
    gammaij_client_ir = log(gammaij_lev_client_ir) - log(gammaij_lev_client_ss);
    
    %% Store results in output matrix
    irs = [A_ir; C_ir; I_ir; Cj_ir; Pj_ir; Ioutj_ir; Moutj_ir; ...
           Lj_ir; Ij_ir; Mj_ir; Yj_ir; Qj_ir; ...
           A_client_ir; Cj_client_ir; Pj_client_ir; Ioutj_client_ir; Moutj_client_ir; ...
           Lj_client_ir; Ij_client_ir; Mj_client_ir; Yj_client_ir; Qj_client_ir; ...
           Kj_ir; GDP_ir; Pmj_client_ir; gammaij_client_ir; C_util_ir];

    %% Full sectoral vectors for Python-side fixed-price aggregation
    pol_c    = (idx.c(1):idx.c(2))       - idx.ss_offset;
    pol_iout = (idx.iout(1):idx.iout(2)) - idx.ss_offset;
    pol_q    = (idx.q(1):idx.q(2))       - idx.ss_offset;
    pol_mout = (idx.mout(1):idx.mout(2)) - idx.ss_offset;

    ps = policies_ss(:);

    sectoral = struct();
    sectoral.C_all    = dynare_simul(idx.c(1):idx.c(2), :)       - ps(pol_c);
    sectoral.Iout_all = dynare_simul(idx.iout(1):idx.iout(2), :) - ps(pol_iout);
    sectoral.Q_all    = dynare_simul(idx.q(1):idx.q(2), :)       - ps(pol_q);
    sectoral.Mout_all = dynare_simul(idx.mout(1):idx.mout(2), :) - ps(pol_mout);
end
