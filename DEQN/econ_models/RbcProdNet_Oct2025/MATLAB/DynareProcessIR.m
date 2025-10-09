function irs = DynareProcessIR(dynare_simul,params,steady_state,labels,parn_sectors,endostates_ss, Cagg_ss, Lagg_ss,policies_ss);
    for idx = 1:numel(labels{1,1})
        sector_idx = labels{1,1}(idx);
        sector_label = labels{1,2}{idx};
        client_idx = labels{1,3}(idx);
        client_label = labels{1,4}{idx};
   
        % Calculate aggregate variables
        A_ir = exp(dynare_simul(parn_sectors + sector_idx, :));
        C_ir = dynare_simul(13 * parn_sectors + 1, :) - log(Cagg_ss);
        L_ir = dynare_simul(13 * parn_sectors + 2, :) - log(Lagg_ss);
        V_ir = dynare_simul(13 * parn_sectors + 6, :);
        Vc_ir = dynare_simul(13 * parn_sectors + 7, :);
        Y_ir = dynare_simul(13 * parn_sectors + 3, :) - steady_state(13 * parn_sectors + 3);
        I_ir = dynare_simul(13 * parn_sectors + 4, :) - steady_state(13 * parn_sectors + 4);
        M_ir = dynare_simul(13 * parn_sectors + 5, :) - steady_state(13 * parn_sectors + 5);
    
        % Calculate sectoral output variables
        Cj_ir = (dynare_simul(2 * parn_sectors + sector_idx, :) - policies_ss(0*parn_sectors+sector_idx));
        Pj_ir = (dynare_simul(10 * parn_sectors + sector_idx, :) - policies_ss(8*parn_sectors+sector_idx));
        Ioutj_ir = (dynare_simul(9 * parn_sectors + sector_idx, :) - policies_ss(7*parn_sectors+sector_idx));
        Moutj_ir = (dynare_simul(7 * parn_sectors + sector_idx, :) - policies_ss(5*parn_sectors+sector_idx));
    
        % Calculate sectoral input variables
        Lj_ir = (dynare_simul(3 * parn_sectors + sector_idx, :) - policies_ss(1*parn_sectors+sector_idx));
        Ij_ir = (dynare_simul(8 * parn_sectors + sector_idx, :) - policies_ss(6*parn_sectors+sector_idx));
        Mj_ir = (dynare_simul(6 * parn_sectors + sector_idx, :) - policies_ss(4*parn_sectors+sector_idx));
        Yj_ir = (dynare_simul(12 * parn_sectors + sector_idx, :) - policies_ss(10*parn_sectors+sector_idx));
        Qj_ir = (dynare_simul(11 * parn_sectors + sector_idx, :) - policies_ss(9*parn_sectors+sector_idx));
        % Kj_ir = exp(dynare_simul(sector_idx, :));
        Kj_ir = (dynare_simul(sector_idx, :) - endostates_ss(sector_idx));
        
        % Calculate client sectoral output variables
        A_client_ir = exp(dynare_simul(parn_sectors + client_idx, :));
        Cj_client_ir = (dynare_simul(2 * parn_sectors + client_idx, :) - policies_ss(0*parn_sectors+client_idx));
        Pj_client_ir = (dynare_simul(10 * parn_sectors + client_idx, :) - policies_ss(8*parn_sectors+client_idx));
        Ioutj_client_ir = (dynare_simul(9 * parn_sectors + client_idx, :) - policies_ss(7*parn_sectors+client_idx));
        Moutj_client_ir = (dynare_simul(7 * parn_sectors + client_idx, :) - policies_ss(5*parn_sectors+client_idx));
    
        % Calculate client sectoral input variables
        Lj_client_ir = (dynare_simul(3 * parn_sectors + client_idx, :) - policies_ss(1*parn_sectors+client_idx));
        Pmj_client_ir = (dynare_simul(5 * parn_sectors + client_idx, :) - policies_ss(3*parn_sectors+client_idx));
        Mj_client_ir = (dynare_simul(6 * parn_sectors + client_idx, :) - policies_ss(4*parn_sectors+client_idx));
        Ij_client_ir = (dynare_simul(8 * parn_sectors + client_idx, :) - policies_ss(6*parn_sectors+client_idx));
        Yj_client_ir = (dynare_simul(12 * parn_sectors + client_idx, :) - policies_ss(10*parn_sectors+client_idx));
        Qj_client_ir = (dynare_simul(11 * parn_sectors + client_idx, :) - policies_ss(9*parn_sectors+client_idx));
        

        % Calculate client expenditure share on affected input
        Pmj_lev_client_ir = exp(dynare_simul(5 * parn_sectors + client_idx, :));
        Pmj_lev_client_ss = exp(policies_ss(3 * parn_sectors + client_idx));
        Pj_lev_ir = exp(dynare_simul(10 * parn_sectors + sector_idx, :));
        Pj_lev_ss = exp(policies_ss(8 * parn_sectors + sector_idx));
        gammaij_lev_client_ir = params.Gamma_M(sector_idx,client_idx).*(Pj_lev_ir./Pmj_lev_client_ir).^(1-params.sigma_m);
        gammaij_lev_client_ss = params.Gamma_M(sector_idx,client_idx)*(Pj_lev_ss/Pmj_lev_client_ss)^(1-params.sigma_m);
        gammaij_client_ir = log(gammaij_lev_client_ir)-log(gammaij_lev_client_ss);
    
        % Store results 
        irs{idx} = [A_ir; C_ir; L_ir; Vc_ir; Cj_ir; Pj_ir; Ioutj_ir; Moutj_ir; Lj_ir; Ij_ir; Mj_ir; Yj_ir; Qj_ir; ...
                            A_client_ir; Cj_client_ir; Pj_client_ir; Ioutj_client_ir; Moutj_client_ir; Lj_client_ir; Ij_client_ir; ...
                            Mj_client_ir; Yj_client_ir; Qj_client_ir; Kj_ir;Y_ir; Pmj_client_ir; gammaij_client_ir];
    end
end