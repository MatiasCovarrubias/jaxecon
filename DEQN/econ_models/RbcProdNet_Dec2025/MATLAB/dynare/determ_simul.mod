// Deterministic simulation of matrix shocksim shocks.
// Requires: shockssim_determ (simul_periods x n_sectors), simul_periods
@#include "model_config.mod"
@#include "ProdNetRbc_base.mod"

initval;
@#for j in 1:n_sectors
    e_@{j}=0;
    a_@{j}=0;
    k_@{j}=k_ss(@{j});
    c_@{j}=policies_ss(@{j});
    l_@{j}=policies_ss(parn_sectors+@{j});
    pk_@{j}=policies_ss(2*parn_sectors+@{j});
    pm_@{j}=policies_ss(3*parn_sectors+@{j});
    m_@{j}=policies_ss(4*parn_sectors+@{j});
    mout_@{j}=policies_ss(5*parn_sectors+@{j});
    i_@{j}=policies_ss(6*parn_sectors+@{j});
    iout_@{j}=policies_ss(7*parn_sectors+@{j});
    p_@{j}=policies_ss(8*parn_sectors+@{j});
    q_@{j}=policies_ss(9*parn_sectors+@{j});
    y_@{j}=policies_ss(10*parn_sectors+@{j});
@#endfor
cagg = policies_ss(11*parn_sectors+1);
lagg = policies_ss(11*parn_sectors+2);
yagg = policies_ss(11*parn_sectors+3);
iagg = policies_ss(11*parn_sectors+4);
magg = policies_ss(11*parn_sectors+5);
        
end;

steady (solve_algo=3);

endval;

@#for j in 1:n_sectors
    e_@{j}=0;
    a_@{j}=0;
    k_@{j}=k_ss(@{j});
    c_@{j}=policies_ss(@{j});
    l_@{j}=policies_ss(parn_sectors+@{j});
    pk_@{j}=policies_ss(2*parn_sectors+@{j});
    pm_@{j}=policies_ss(3*parn_sectors+@{j});
    m_@{j}=policies_ss(4*parn_sectors+@{j});
    mout_@{j}=policies_ss(5*parn_sectors+@{j});
    i_@{j}=policies_ss(6*parn_sectors+@{j});
    iout_@{j}=policies_ss(7*parn_sectors+@{j});
    p_@{j}=policies_ss(8*parn_sectors+@{j});
    q_@{j}=policies_ss(9*parn_sectors+@{j});
    y_@{j}=policies_ss(10*parn_sectors+@{j});
@#endfor
cagg = policies_ss(11*parn_sectors+1);
lagg = policies_ss(11*parn_sectors+2);
yagg = policies_ss(11*parn_sectors+3);
iagg = policies_ss(11*parn_sectors+4);
magg = policies_ss(11*parn_sectors+5);
        
end;

// shockssim_determ must be (500 x n_sectors) 
// Row 1: initial period, Rows 2:499: simulation, Row 500: terminal
// periods = simul_T_determ - 2 = 500 - 2 = 498
perfect_foresight_setup(periods=498);
oo_.exo_simul = shockssim_determ;
perfect_foresight_solver(tolf=1e-3);


