# Differentiable Communication Games Origin

_(v 0.1 — July 2025)_

---

## 1 Genesis

Differentiable physics engines (Brax, MuJoCo‐AD, JAX MD) proved that giving agents _analytic gradients_ through their environment reduces sample complexity and stabilises learning.

In the multi agent case, If in the ‘environment’ we have differentiable rewards, transitions paths, and policy functions (e.g. neural nets) of other agents, we can keep the whole interaction loop differentiable—_provided the communication protocol itself is designed for gradient flow._

That realisation launched the **Differentiable Communication Games (DCG)** project: co-design a language protocol _and_ a neural architecture that maximise the utility of pathwise gradients between agents.

---

## 2 Project Object(ive)

> Co-design a _surface language_ **Lᴰ** and a _Transformer-family architecture_ **Tᴳ** such that  
> gradients ∂r/∂m propagate with minimal variance from any receiver to any sender participating in a multi-agent game.

Concretely, we aim to:

1. Replace REINFORCE-style language RL with low-variance path-wise gradients.
2. Think about how rational inattention can regulate the design of both the attention unit and the language. For example, it can regulate how to propagate gradients. You can constrain the amount of gradient propagation flops. For example, you can choose to ignore the second order effects on irrelevant players.
3. Produce protocols interpretable enough to audit or discretize _after_ training, but fully continuous during optimization.

---

## 3 Foundational Principles

| #   | Principle                        | One-line Rationale                                                                              |
| --- | -------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | **End-to-End Differentiability** | Every computational block between action and reward must expose Jacobians.                      |
| 2   | **Co-Design**                    | Optimise architecture **and** language jointly; sub-optimality in either leaks gradient signal. |
| 3   | **Rational Inattention**         | Treat gradient bandwidth as a priced resource; agents pay KL(cost) for every communicated bit.  |

---

## 4 Technical Pillars

1. **Gradient-Aware Transformers (Tᴳ)**  
   • Attention scores modulated by downstream gradient norms.  
   • Bottleneck layers enforce information budgets.

2. **Differentiable Language Lᴰ**  
   • Messages are ℝ^{L×d} tensors during training, optionally vector-quantised post-hoc.  
   • Soft STOP symbol and annealed temperature avoid hard arg-max.

3. **Auto-Gradient Channels**  
   • Library hooks automatically expose ∂loss/∂message to the sender—no manual plumbing.

4. **Information-Theoretic Regularisers**  
   • KL(msg ‖ 𝓝(0,I)) and Fisher-rank penalties maintain concise, non-redundant protocols.

---

## 5 Game-Theoretic & Economic Lens

• **Strategic Reasoning:** Agents learn not only best responses but _Jacobian responses_—how their message tweaks the partner’s policy surface.  
• **Bargaining & Public-Goods Games:** Measure welfare gains vs. classic discrete-chat counterparts.  
• **Kernel-as-Message:** Passing an entire function (e.g. demand curve) in one differentiable shot; credit back-propagates through closed-form regression.

Possible applications:

-   Dynamic oligopolistic games with imperfect information.
-   Information sharing between firms (would you give up your productivity signal?).
-   Bargaining a la Rubinstein
-   Function-as-Message (think supply curve submission, Kyle model, etc.)

---

## 6 Roadmap

| Month | Milestone                                                                                                | Key Metrics                         |
| ----- | -------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1-2   | Build DCG-lib in JAX with gradient taps & info bottlenecks.                                              | Unit tests; GSNR↑ vs. REINFORCE.    |
| 2-3   | Continuous referential game baseline.                                                                    | Sample efficiency, final accuracy.  |
| 3-4   | Implement Gradient-Aware Attention layer; ablate.                                                        | Convergence speed, info-cost curve. |
| 5-6   | Double-blind bargaining demo + pre-print _“Differentiable Transformers for Rational Inattention Games”_. | Surplus vs. Nash, reproducibility.  |

---

## 7 Speculative Discussions

The following topics outline forward-looking questions that blend information theory, game theory, and differentiable multi-agent RL. They are not yet on the core roadmap but may crystallise into future workstreams:

1. **Messaging Entire Policies:** What is the rate–distortion frontier for transmitting a neural-network policy within a fixed bit-budget? How should information be prioritised (e.g. low-rank factors, Taylor coefficients, gradients)?
2. **Cost of Lying & Trust Formation:** If an agent intentionally misreports its policy or gradients, how does the extra coding cost scale with deviation from truth? Can receivers detect deception by exploiting compression inefficiencies?
3. **Gradient-Aware Bottlenecks:** Allocate limited channel capacity to those Jacobian components that most affect downstream loss, effectively sending a compressed sensitivity map.
4. **Progressive / Anytime Codes:** Design encodings where early bytes convey coarse policy shape and later bytes refine precision—enabling interruption-tolerant coordination.
5. **Alignment Signals vs. Cheap Talk:** Model how receivers update beliefs about sender alignment given both observed actions and communicated self-reports, under bandwidth constraints.
6. **Reputation & Strategic Disclosure:** Study equilibria where revealing more of one’s policy today improves future cooperation but exposes exploitable structure. Map phase diagrams of silence, partial, and full disclosure regimes.

---

## 8 The Witsenhausen Challenge: AlphaWits

**Target:** Improve upon the [Witsenhausen counterexample](https://doi.org/10.1137/0306011)—a 55-year-old open problem that sits at the intersection of control theory, information theory, and game theory.

### The Problem

The Witsenhausen counterexample, first posed in 1968, is a deceptively simple two-stage control problem that violated classical assumptions about optimal control:

• **Initial state** x₀ ~ 𝒩(0, σ²) observed by Controller 1  
• **Stage-1 action** u₁ = π₁(x₀) (simultaneously control action _and_ message)  
• **Observation** y = x₀ + u₁ + w received by Controller 2, where w ~ 𝒩(0, 1)  
• **Stage-2 action** u₂ = π₂(y)  
• **Joint cost** L = k²u₁² + (x₀ + u₁ - u₂)²

The counterintuitive result: despite shared objectives, **linear policies are not optimal**—contradicting classical LQG theory. The true global optimum remains unknown for most parameter values.

### Why This Is the Perfect DCG Test-Bed

1. **Communication ≡ Control:** Action u₁ simultaneously moves the system _and_ encodes information—exactly the "action = message" philosophy of DCG.

2. **Non-Convex with Known Bounds:** Rate-distortion theory provides fundamental lower bounds; existing heuristics give upper bounds. Your algorithms can be rigorously benchmarked.

3. **Differentiable by Design:** Both controllers share the same cost function, making gradient propagation through u₁ mathematically natural.

4. **Scalable Research Program:** Success unlocks multi-dimensional generalizations, bandwidth constraints, strategic (non-aligned) variants, and progressive coding schemes.

### Research Angles

• **Rate-Distortion vs. Gradient Flow:** How close can learned policies get to Shannon's theoretical limit?  
• **Neural vs. Piece-wise Linear:** Does a Transformer bottleneck naturally rediscover the known step-function structure?  
• **Trust & Deception:** Allow misaligned objectives and measure welfare loss or detectability of lies.  
• **Progressive Codes:** Add stages to test "anytime" messaging from your Speculative Discussions.

### Key References

• Witsenhausen, H.S. ["A counterexample in stochastic optimum control."](https://doi.org/10.1137/0306011) _SIAM Journal on Control_, 6(1):131-147, 1968.  
• Ho, Y.C. & Chu, K.C. ["Team decision theory and information structures."](https://doi.org/10.1109/TAC.1972.1099829) _IEEE Trans. Automatic Control_, 17(1):15-22, 1972.  
• Witsenhausen, H.S. ["The intrinsic model for discrete stochastic control."](https://doi.org/10.1007/978-3-642-46317-4_24) In _Control Theory and Computer Systems Modelling_, Springer, 1975.

---

<!-- ---
