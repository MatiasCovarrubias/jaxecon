#!/usr/bin/env python3
"""
Quick diagnostic script to identify NaN sources when changing volatility.
Run from repo root: python -m DEQN.econ_models.RbcProdNet_Dec2025.diagnose_nan
"""

import os
import sys

# Setup path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import jax
import jax.numpy as jnp
import jax.random as random
import scipy.io as sio
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_debug_nans", True)

# Import the models
from DEQN.econ_models.RbcProdNet_Dec2025.model import Model
from DEQN.neural_nets.with_loglinear_baseline import NeuralNet


def main():
    print("=" * 70)
    print("NaN DIAGNOSTIC SCRIPT")
    print("=" * 70)
    
    # Load model data
    model_dir = os.path.join(repo_root, "DEQN/econ_models/RbcProdNet_Dec2025")
    model_path = os.path.join(model_dir, "ModelData.mat")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model_data.mat")
    
    print(f"Loading model from: {model_path}")
    model_data = sio.loadmat(model_path, simplify_cells=True)
    
    precision = jnp.float64
    
    # Extract data (handle both structures)
    if "ModelData" in model_data:
        md = model_data["ModelData"]
        n_sectors = md["SteadyState"]["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(md["SteadyState"]["endostates_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params = md["SteadyState"]["parameters"].copy()
        state_sd = jnp.array(md["Simulation"]["states_sd"], dtype=precision)
        policies_sd = jnp.array(md["Simulation"]["policies_sd"], dtype=precision)
        policies_ss = jnp.array(md["SteadyState"]["policies_ss"], dtype=precision)
        C_matrix = md["Solution"]["StateSpace"]["C"]
    else:
        soldata = model_data["SolData"]
        n_sectors = soldata["parameters"]["parn_sectors"]
        a_ss = jnp.zeros(shape=(n_sectors,), dtype=precision)
        k_ss = jnp.array(soldata["k_ss"], dtype=precision)
        state_ss = jnp.concatenate([k_ss, a_ss])
        params = soldata["parameters"].copy()
        state_sd = jnp.array(soldata["states_sd"], dtype=precision)
        policies_sd = jnp.array(soldata["policies_sd"], dtype=precision)
        policies_ss = jnp.array(soldata["policies_ss"], dtype=precision)
        C_matrix = soldata["C"]
    
    print(f"n_sectors: {n_sectors}")
    print(f"state_ss shape: {state_ss.shape}")
    print(f"policies_ss shape: {policies_ss.shape}")
    
    # Test different volatility scales
    volatility_scales = [0.1, 0.2, 0.5, 1.0]
    
    for vol_scale in volatility_scales:
        print(f"\n{'='*70}")
        print(f"TESTING VOLATILITY SCALE: {vol_scale}")
        print(f"{'='*70}")
        
        # Create model with this volatility
        econ_model = Model(
            parameters=params,
            state_ss=state_ss,
            policies_ss=policies_ss,
            state_sd=state_sd,
            policies_sd=policies_sd,
            double_precision=True,
            volatility_scale=vol_scale,
        )
        
        # Create neural network
        dim_policies = econ_model.dim_policies
        neural_net = NeuralNet(
            features=[32, 32, dim_policies],
            C=C_matrix,
            states_sd=state_sd,
            policies_sd=policies_sd,
            param_dtype=precision,
        )
        
        # Initialize network
        rng = random.PRNGKey(42)
        rng, init_rng, state_rng, mc_rng = random.split(rng, 4)
        
        dummy_state = jnp.zeros(econ_model.dim_states)
        params_nn = neural_net.init(init_rng, dummy_state)
        
        # Test 1: Check MC shocks magnitude
        mc_shocks = econ_model.mc_shocks(mc_rng, mc_draws=32)
        print(f"\nMC Shocks stats:")
        print(f"  Shape: {mc_shocks.shape}")
        print(f"  Min: {jnp.min(mc_shocks):.4f}, Max: {jnp.max(mc_shocks):.4f}")
        print(f"  Mean: {jnp.mean(mc_shocks):.4f}, Std: {jnp.std(mc_shocks):.4f}")
        
        # Test 2: Get initial state and policy
        init_state = econ_model.initial_state(state_rng, range=6)
        init_policy = neural_net.apply(params_nn, init_state)
        
        print(f"\nInitial state z-score stats:")
        print(f"  Min: {jnp.min(init_state):.4f}, Max: {jnp.max(init_state):.4f}")
        
        print(f"\nInitial policy z-score stats:")
        print(f"  Min: {jnp.min(init_policy):.4f}, Max: {jnp.max(init_policy):.4f}")
        
        # Test 3: Step with each MC shock and check next states
        print(f"\nTesting state transitions with MC shocks...")
        has_issues = False
        
        for i, shock in enumerate(mc_shocks):
            next_state = econ_model.step(init_state, init_policy, shock)
            
            if jnp.any(jnp.isnan(next_state)) or jnp.any(jnp.isinf(next_state)):
                print(f"  Shock {i}: NaN/Inf in next_state!")
                has_issues = True
                continue
            
            # Check how far from steady state this puts us
            state_notnorm = next_state * state_sd + state_ss
            a_next = state_notnorm[n_sectors:]
            
            if i < 3:  # Print first few
                print(f"  Shock {i}: a_next range [{jnp.min(a_next):.4f}, {jnp.max(a_next):.4f}]")
        
        # Test 4: Check policies at next states
        print(f"\nTesting policies at next states...")
        
        def test_single_shock(shock):
            next_state = econ_model.step(init_state, init_policy, shock)
            next_policy = neural_net.apply(params_nn, next_state)
            
            # Check policy in levels
            policy_notnorm = next_policy * policies_sd + policies_ss
            policy_levels = jnp.exp(policy_notnorm)
            
            # Check specific problematic terms
            Cagg = policy_levels[11 * n_sectors]
            Lagg = policy_levels[11 * n_sectors + 1]
            
            # The critical term in MgUtCagg
            theta = params["partheta"]
            eps_l = params["pareps_l"]
            MgUtCagg_inner = Cagg - theta / (1 + eps_l ** (-1)) * Lagg ** (1 + eps_l ** (-1))
            
            return {
                "Cagg": Cagg,
                "Lagg": Lagg,
                "MgUtCagg_inner": MgUtCagg_inner,
                "policy_min": jnp.min(policy_levels),
                "policy_max": jnp.max(policy_levels),
            }
        
        # Test on all MC shocks
        results = jax.vmap(test_single_shock)(mc_shocks)
        
        print(f"\nPolicy diagnostics across all MC shocks:")
        print(f"  Cagg range: [{jnp.min(results['Cagg']):.6f}, {jnp.max(results['Cagg']):.6f}]")
        print(f"  Lagg range: [{jnp.min(results['Lagg']):.6f}, {jnp.max(results['Lagg']):.6f}]")
        print(f"  MgUtCagg_inner range: [{jnp.min(results['MgUtCagg_inner']):.6f}, {jnp.max(results['MgUtCagg_inner']):.6f}]")
        print(f"  Policy levels range: [{jnp.min(results['policy_min']):.6f}, {jnp.max(results['policy_max']):.6f}]")
        
        if jnp.any(results['MgUtCagg_inner'] <= 0):
            print(f"  ⚠️  WARNING: MgUtCagg_inner becomes non-positive! This causes NaN!")
            n_negative = jnp.sum(results['MgUtCagg_inner'] <= 0)
            print(f"      {n_negative} out of {len(mc_shocks)} MC draws have issues")
        
        # Test 5: Try to compute expectations
        print(f"\nTesting expect_realization...")
        try:
            def compute_expect_for_shock(shock):
                next_state = econ_model.step(init_state, init_policy, shock)
                next_policy = neural_net.apply(params_nn, next_state)
                return econ_model.expect_realization(next_state, next_policy)
            
            expect_all = jax.vmap(compute_expect_for_shock)(mc_shocks)
            expect_mean = jnp.mean(expect_all, axis=0)
            
            if jnp.any(jnp.isnan(expect_mean)):
                print(f"  ⚠️  NaN in expectation!")
                print(f"  NaN count: {jnp.sum(jnp.isnan(expect_all))}")
            else:
                print(f"  Expectation computed successfully")
                print(f"  Mean expect range: [{jnp.min(expect_mean):.6f}, {jnp.max(expect_mean):.6f}]")
        except Exception as e:
            print(f"  ⚠️  Exception in expect_realization: {e}")
        
        # Test 6: Try to compute loss
        print(f"\nTesting loss computation...")
        try:
            expect_mean = jnp.mean(jax.vmap(compute_expect_for_shock)(mc_shocks), axis=0)
            loss_result = econ_model.loss(init_state, expect_mean, init_policy)
            mean_loss = loss_result[0]
            
            if jnp.isnan(mean_loss):
                print(f"  ⚠️  NaN in loss!")
            else:
                print(f"  Loss computed successfully: {mean_loss:.6f}")
        except Exception as e:
            print(f"  ⚠️  Exception in loss: {e}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nSUMMARY:")
    print("- If MgUtCagg_inner becomes non-positive at higher volatility,")
    print("  that's the source of your NaN (negative number to fractional power)")
    print("- The fix is either:")
    print("  1. Clip the inner term to be always positive")
    print("  2. Use a different utility parameterization")
    print("  3. Warm-start training at low volatility before increasing")


if __name__ == "__main__":
    main()

