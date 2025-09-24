Work in progress.

## Description

This repository contains a collection of solution algorithms for dynamic economic models. They are implemented in Python using the Jax library for Scientific Computing and Machine Learning.

## Implementation in Google Colab

-   The algorithms are implemented as self-contained google colab notebooks and also as a modular python code.
-   They can be run using CPU, GPU or TPU in Google Colab, by changing 1 line of code.
-   If you subscribe to Google Colab Pro (~$10/month and ~$50/month tiers are available), you can choose between 2 GPU tiers, TPU, and choose whether you need extra RAM.
-   With the free Colab, you can still run these with no problem on a CPU or a (low tier) GPU.

## Algorithms (list is growing)

1. VFI: Heavily parallelized Value Function Iteration.

In this implementation I compare different parallelization at different scales, for both GPU and TPU.
I find that both TPUs and GPUs can handle very large scales with the correct parallelization, but TPUs dominate at some point.

2. Jax-DEQN

An implementation using Jax of the Deep Equilibrium Network (DEQN) in Jax adapted for continuous shocks instead.
The implementation allows the user to input all the model logic in one simple class.
All the version can be run in CPU, GPU and TPU backends.

3. Analytical Policy Gradient (APG): A policy gradient algorithm for training neural networks to sole Markov Decision Processes (MDPs) and Strategic Games (SG). The key feature of this algorithm is that ist uses the fact that environmnets have differentiable spet and reward functions, so we can use those function to calculate the policy gradient.
