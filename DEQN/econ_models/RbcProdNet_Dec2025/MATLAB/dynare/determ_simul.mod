// Deterministic simulation of matrix shocksim shocks.
// Requires: shockssim_determ (T_total x n_sectors)
//
// BURN-IN/BURN-OUT APPROACH:
// - Total periods = burn_in + T_active + burn_out = 100 + 500 + 100 = 700
// - Burn-in/burn-out periods have ZERO shocks (smooth transition from/to SS)
// - Active shocks placed in middle period (rows burn_in+2 to burn_in+1+T_active)
// - This helps convergence for longer simulations by relaxing boundary constraints
//
// Shock matrix structure (T_total=700 rows):
//   Row 1: initial period (t=0), zero shock
//   Rows 2-101: burn-in (t=1 to 100), zero shocks
//   Rows 102-601: active (t=101 to 600), random shocks
//   Rows 602-699: burn-out (t=601 to 698), zero shocks
//   Row 700: terminal period (t=699), zero shock
//
// Dynare periods = T_total - 2 = 700 - 2 = 698
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

// periods = T_total - 2 = 700 - 2 = 698
// IMPORTANT: Must hardcode this value (Dynare macro limitation)
// If changing burn_in/burn_out/simul_T_pf in main.m, update this!
perfect_foresight_setup(periods=698);
oo_.exo_simul = shockssim_determ;
perfect_foresight_solver(endogenous_terminal_period, tolf=1e-3);


