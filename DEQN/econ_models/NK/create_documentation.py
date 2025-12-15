#!/usr/bin/env python3
"""
Generate PDF documentation for the Three-Equation New Keynesian Model.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False  # Use mathtext instead of LaTeX

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_documentation_pdf(save_path):
    """Create a PDF document explaining the NK model."""
    
    with PdfPages(save_path) as pdf:
        
        # =====================================================================
        # PAGE 1: Title and Introduction
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        
        y = 0.95
        
        # Title
        ax.text(0.5, y, "Three-Equation New Keynesian Model", 
                fontsize=20, fontweight='bold', ha='center', va='top')
        y -= 0.06
        ax.text(0.5, y, "Based on Galí (2015), Chapter 3", 
                fontsize=14, style='italic', ha='center', va='top')
        y -= 0.03
        ax.text(0.5, y, "Implementation in JAXecon/DEQN", 
                fontsize=12, ha='center', va='top', color='#555555')
        
        y -= 0.08
        ax.axhline(y=y, xmin=0.1, xmax=0.9, color='black', linewidth=0.5)
        
        # Introduction
        y -= 0.05
        ax.text(0.0, y, "1. Introduction", fontsize=14, fontweight='bold', va='top')
        y -= 0.04
        
        intro_text = """The three-equation New Keynesian (NK) model is the workhorse framework for 
monetary policy analysis. It combines optimizing behavior by households and firms 
with nominal rigidities (sticky prices) to generate a role for monetary policy.

The model consists of three key equations:
  • Dynamic IS Equation (demand side)
  • New Keynesian Phillips Curve (supply side)  
  • Monetary Policy Rule (Taylor rule)

Unlike Real Business Cycle models, the NK model is purely forward-looking with no 
endogenous state variables. The only state variables are exogenous shock processes."""
        
        ax.text(0.0, y, intro_text, fontsize=10, va='top', 
                family='serif', linespacing=1.5)
        
        y -= 0.28
        
        # Key features
        ax.text(0.0, y, "Key Features:", fontsize=11, fontweight='bold', va='top')
        y -= 0.04
        
        features = """  • Forward-looking expectations (rational expectations)
  • Nominal price rigidities (Calvo pricing)
  • Monetary policy affects real variables in the short run
  • Divine coincidence: stabilizing inflation also stabilizes output gap
  • Steady state: zero inflation, zero output gap"""
        
        ax.text(0.0, y, features, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # PAGE 2: The Three Equations
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        
        y = 0.95
        
        ax.text(0.0, y, "2. The Three Equations", fontsize=14, fontweight='bold', va='top')
        y -= 0.06
        
        # Equation 1: NKPC
        ax.text(0.0, y, "2.1 New Keynesian Phillips Curve (NKPC)", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.04
        
        ax.text(0.5, y, r"$\pi_t = \beta \, \mathbb{E}_t\{\pi_{t+1}\} + \kappa \, \tilde{y}_t$", 
                fontsize=14, ha='center', va='top')
        y -= 0.04
        ax.text(0.5, y, "(Galí Equation 21)", fontsize=9, ha='center', va='top', color='gray')
        y -= 0.04
        
        nkpc_text = """where:
  • πₜ = inflation rate (log deviation from steady state)
  • ỹₜ = output gap (log deviation of output from natural level)
  • β = household discount factor (0.99 quarterly)
  • κ = slope of Phillips curve, depends on price stickiness

The NKPC relates current inflation to expected future inflation and the output gap.
Higher output gap (demand pressure) leads to higher inflation."""
        ax.text(0.0, y, nkpc_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        y -= 0.22
        
        # Equation 2: DIS
        ax.text(0.0, y, "2.2 Dynamic IS Equation (DIS)", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.04
        
        ax.text(0.5, y, r"$\tilde{y}_t = \mathbb{E}_t\{\tilde{y}_{t+1}\} - \frac{1}{\sigma}(i_t - \mathbb{E}_t\{\pi_{t+1}\} - r_t^n)$", 
                fontsize=14, ha='center', va='top')
        y -= 0.04
        ax.text(0.5, y, "(Galí Equation 22)", fontsize=9, ha='center', va='top', color='gray')
        y -= 0.04
        
        dis_text = """where:
  • iₜ = nominal interest rate
  • rₜⁿ = natural rate of interest
  • σ = inverse elasticity of intertemporal substitution (CRRA)

The DIS is derived from the household's Euler equation. Higher real interest rates
(iₜ - 𝔼ₜ{πₜ₊₁}) relative to the natural rate reduce current output gap."""
        ax.text(0.0, y, dis_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        y -= 0.20
        
        # Equation 3: Taylor Rule
        ax.text(0.0, y, "2.3 Monetary Policy Rule (Taylor Rule)", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.04
        
        ax.text(0.5, y, r"$i_t = \rho + \phi_\pi \pi_t + \phi_y \tilde{y}_t + v_t$", 
                fontsize=14, ha='center', va='top')
        y -= 0.04
        ax.text(0.5, y, "(Galí Equation 25)", fontsize=9, ha='center', va='top', color='gray')
        y -= 0.04
        
        taylor_text = """where:
  • ρ = steady-state real interest rate (= -log(β))
  • φπ = response to inflation (typically 1.5, Taylor principle: φπ > 1)
  • φy = response to output gap (typically 0.5/4 = 0.125 quarterly)
  • vₜ = monetary policy shock (deviation from rule)

The Taylor rule describes how the central bank sets interest rates in response
to inflation and output gap deviations from target."""
        ax.text(0.0, y, taylor_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # PAGE 3: Natural Rate and Shock Processes
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        
        y = 0.95
        
        ax.text(0.0, y, "3. Natural Rate and Shock Processes", fontsize=14, fontweight='bold', va='top')
        y -= 0.06
        
        # Natural rate
        ax.text(0.0, y, "3.1 Natural Rate of Interest", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.04
        
        ax.text(0.5, y, r"$r_t^n = \rho + \sigma \psi_{ya}^n \, \mathbb{E}_t\{\Delta a_{t+1}\}$", 
                fontsize=14, ha='center', va='top')
        y -= 0.04
        ax.text(0.5, y, "(Galí Equation 23)", fontsize=9, ha='center', va='top', color='gray')
        y -= 0.04
        
        rn_text = """The natural rate is the real interest rate that would prevail under flexible prices.
For an AR(1) productivity process aₜ = ρₐ aₜ₋₁ + εₜᵃ:

    𝔼ₜ{Δaₜ₊₁} = 𝔼ₜ{aₜ₊₁ - aₜ} = (ρₐ - 1) aₜ

So in deviations from steady state:  rₜⁿ - ρ = σ ψⁿya (ρₐ - 1) aₜ

Since ρₐ < 1, a positive productivity shock lowers the natural rate."""
        ax.text(0.0, y, rn_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        y -= 0.22
        
        # Shock processes
        ax.text(0.0, y, "3.2 Exogenous Shock Processes", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.05
        
        shock_text = """The model has two exogenous AR(1) shock processes:

Productivity Shock (affects natural rate):
    aₜ = ρₐ aₜ₋₁ + σₐ εₜᵃ,    εₜᵃ ~ N(0,1)
    
Monetary Policy Shock (deviation from Taylor rule):
    vₜ = ρᵥ vₜ₋₁ + σᵥ εₜᵛ,    εₜᵛ ~ N(0,1)

Default calibration:
  • ρₐ = 0.9  (persistent productivity)
  • ρᵥ = 0.5  (less persistent monetary shock)
  • σₐ = 0.01 (1% std dev)
  • σᵥ = 0.0025 (25 basis points)"""
        ax.text(0.0, y, shock_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        y -= 0.32
        
        # State space
        ax.text(0.0, y, "3.3 State Space Representation", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.05
        
        ss_text = """State Variables (exogenous):     sₜ = [aₜ, vₜ]
Policy Variables (endogenous):   pₜ = [ỹₜ, πₜ]

The interest rate iₜ is determined by the Taylor rule given the policy variables.
The model is solved by finding a policy function p(s) that satisfies the 
equilibrium conditions (Euler equations) at all points in the state space."""
        ax.text(0.0, y, ss_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # PAGE 4: Implementation
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        
        y = 0.95
        
        ax.text(0.0, y, "4. Implementation in JAXecon/DEQN", fontsize=14, fontweight='bold', va='top')
        y -= 0.06
        
        ax.text(0.0, y, "4.1 File Structure", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.04
        
        files_text = """DEQN/econ_models/NK/
├── __init__.py       # Module exports
├── model.py          # Model class with equilibrium conditions
├── train.py          # Training script
└── analysis.py       # Impulse response analysis"""
        ax.text(0.0, y, files_text, fontsize=10, va='top', family='monospace', linespacing=1.4)
        
        y -= 0.14
        
        ax.text(0.0, y, "4.2 Key Methods in model.py", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.05
        
        methods_text = """• __init__(): Initialize parameters (β, σ, κ, φπ, φy, ρₐ, ρᵥ, etc.)

• step(state, policy, shock): Transition function
    Returns next state: sₜ₊₁ = [ρₐ aₜ + σₐ εₜᵃ, ρᵥ vₜ + σᵥ εₜᵛ]

• expect_realization(state_next, policy_next): 
    Returns [ỹₜ₊₁, πₜ₊₁] for computing expectations

• loss(state, expect, policy): Euler equation residuals
    Computes residuals for DIS and NKPC:
    
    DIS residual:  ỹₜ - 𝔼ₜ{ỹₜ₊₁} + (1/σ)(iₜ - 𝔼ₜ{πₜ₊₁} - rₜⁿ)
    NKPC residual: πₜ - β 𝔼ₜ{πₜ₊₁} - κ ỹₜ
    
    where iₜ = φπ πₜ + φy ỹₜ + vₜ (Taylor rule)

• get_aggregates(): Compute all variables from states and policies"""
        ax.text(0.0, y, methods_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        y -= 0.38
        
        ax.text(0.0, y, "4.3 DEQN Algorithm", 
                fontsize=12, fontweight='bold', va='top', color='#2E86AB')
        y -= 0.05
        
        algo_text = """The Deep Equilibrium Network (DEQN) algorithm:

1. Neural network πθ(s) maps states to policies: [aₜ, vₜ] → [ỹₜ, πₜ]

2. Simulate episodes using the policy network

3. Compute Euler equation residuals using Monte Carlo expectations

4. Minimize squared residuals via gradient descent (Adam optimizer)

5. Repeat until convergence (accuracy > 99%)

The trained network provides an approximate global solution to the model."""
        ax.text(0.0, y, algo_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # PAGE 5: Calibration and Usage
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis('off')
        
        y = 0.95
        
        ax.text(0.0, y, "5. Calibration", fontsize=14, fontweight='bold', va='top')
        y -= 0.06
        
        calib_text = """Default calibration (quarterly frequency):

Parameter    Value    Description
─────────────────────────────────────────────────────
β            0.99     Discount factor
σ            1.0      CRRA coefficient (log utility)
κ            0.1275   NKPC slope
φπ           1.5      Taylor rule: inflation response
φy           0.125    Taylor rule: output gap response (0.5/4)
ρₐ           0.9      Productivity shock persistence
ρᵥ           0.5      Monetary shock persistence
σₐ           0.01     Productivity shock std dev
σᵥ           0.0025   Monetary shock std dev
ψⁿya         1.0      Natural output elasticity to productivity"""
        ax.text(0.0, y, calib_text, fontsize=10, va='top', family='monospace', linespacing=1.3)
        
        y -= 0.34
        
        ax.text(0.0, y, "6. Usage", fontsize=14, fontweight='bold', va='top')
        y -= 0.05
        
        usage_text = """Training the model:
    python -m DEQN.econ_models.NK.train

Running impulse response analysis:
    python -m DEQN.econ_models.NK.analysis

Using the model in Python:
    from DEQN.econ_models.NK.model import Model
    
    # Create model with default parameters
    model = Model()
    
    # Or with custom calibration
    model = Model(beta=0.99, kappa=0.15, phi_pi=2.0)"""
        ax.text(0.0, y, usage_text, fontsize=10, va='top', family='monospace', linespacing=1.3)
        
        y -= 0.26
        
        ax.text(0.0, y, "7. References", fontsize=14, fontweight='bold', va='top')
        y -= 0.05
        
        refs_text = """• Galí, J. (2015). Monetary Policy, Inflation, and the Business Cycle: 
  An Introduction to the New Keynesian Framework. Princeton University Press.
  Chapter 3: The Basic New Keynesian Model.

• Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep Equilibrium Nets.
  International Economic Review, 63(4), 1471-1525."""
        ax.text(0.0, y, refs_text, fontsize=10, va='top', family='serif', linespacing=1.4)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Documentation saved to: {save_path}")


if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "NK_Model_Documentation.pdf")
    
    create_documentation_pdf(save_path)

