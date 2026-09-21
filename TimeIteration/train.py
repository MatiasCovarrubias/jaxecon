#!/usr/bin/env python3
"""Smoke run for TimeIteration: closed-form gate, then irreversible RBC."""

import os
import sys

if __package__ in {None, ""}:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import jax
import jax.numpy as jnp

from TimeIteration.algorithm.solve import solve
from TimeIteration.diagnostics import allocation_on_grid, ergodic_stats, simulate
from TimeIteration.models.protocol import GridSpec
from TimeIteration.models.rbc import RbcModel, closed_form_params, default_params


def _print_closed_form():
    model = RbcModel(unknown="saving_rate")
    params = closed_form_params()
    spec = GridSpec(n_a=5, n_k=21, k_min_rel=0.75, k_max_rel=1.3)
    sol = solve(model, params, spec, tol=1e-12, max_iter=80)
    saving = allocation_on_grid(model, params, sol.grids, sol.x_grid).s
    target = float(params.alpha * params.beta)
    max_err = float(jnp.max(jnp.abs(saving - target)))
    print("Closed form (log utility, delta=1)")
    print(f"  alpha*beta           = {target:.12f}")
    print(f"  max |s - alpha*beta| = {max_err:.3e}")
    print(f"  residual norm        = {float(sol.info.residual_norm):.3e}")
    print(f"  iterations           = {int(sol.info.iterations)}")
    return max_err


def _print_irreversible():
    model = RbcModel(irreversible=True)
    params = default_params(i_min_frac=0.975, phi=0.0, sigma=2.0)
    spec = GridSpec(n_a=5, n_k=31, k_min_rel=0.85, k_max_rel=1.2)
    sol = solve(model, params, spec, tol=1e-9, max_iter=40, newton_steps=15, anderson_memory=5)
    aux = allocation_on_grid(model, params, sol.grids, sol.x_grid)
    binds = aux.I <= params.i_min_frac * sol.grids.ss.I + 1e-8
    traj = simulate(model, params, sol.grids, sol.x_grid, jax.random.PRNGKey(0), T=128, N=32)
    stats = ergodic_stats(params, sol.grids, traj)
    print("Irreversible RBC (I >= 0.975 I_ss)")
    print(f"  residual norm        = {float(sol.info.residual_norm):.3e}")
    print(f"  iterations           = {int(sol.info.iterations)}")
    print(f"  grid bind share      = {float(jnp.mean(binds)):.3f}")
    print(f"  simul K/Kss          = {float(stats.K_rel):.4f}")
    print(f"  simul bind share     = {float(stats.bind_frac):.3f}")
    print(f"  simul mean s         = {float(stats.s_mean):.4f}")


def main():
    print("TimeIteration smoke")
    max_err = _print_closed_form()
    _print_irreversible()
    if max_err > 1e-10:
        raise SystemExit(f"closed-form gate failed: max |s - alpha*beta| = {max_err:.3e}")
    print("ok")


if __name__ == "__main__":
    main()
