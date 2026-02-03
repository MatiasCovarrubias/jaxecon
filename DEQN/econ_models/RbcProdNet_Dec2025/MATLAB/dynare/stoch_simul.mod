// First-Order (Linear) Approximation Solution
// 
// This file computes the first-order perturbation solution.
// Outputs: ghx (state transition), ghu (shock impact) → A, B, C, D matrices
// Use stoch_simul_2ndOrder.mod for second-order approximation.

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

shocks;
@#for j in 1:n_sectors
   var e_@{j} = parSigma_A(@{j},@{j});
@#endfor
@#for j in 1:n_sectors-1
    @#for i in j+1:n_sectors
        var e_@{j}, e_@{i} = parSigma_A(@{i},@{j});
    @#endfor
@#endfor
end;

// First-order approximation: computes ghx, ghu (linear policy functions)
// periods=0: no simulation (we do that in MATLAB with simult_)
// ar=1: compute first autocorrelation
stoch_simul(order=1, periods=0, irf=0, ar=1, nocorr, nograph, nofunctions) cagg lagg yagg iagg magg;