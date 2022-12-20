Work in progress.

## Description
This repository contains a collection of solution algorithms for dynamic models. 

## Implementation in Google Colab
- The algorithms are implemented as self-contained google colab notebooks. 
- THey can be run using CPU, GPU or TPU in Google Colab, by changing 1 line of code. 
- If you subscribe to Google Colab Pro (~$10/month and ~$50/month tiers are available), you can choose between 2 GPU tiers, TPU, and choose wether you need extra RAM.
- With the free Colab, you can still run these with no problem on a CPU. Furthermore, most of the time you can get a basic tier GPU.
-  
## Algorithms (list is growing)

1. VFI: Heavily parallelized Value Function Iteration.

In this implementation I compare different parallelization at different scales, for both GPU and TPU. 
I find that both TPUs and GPUs can handle very large scales with the correct parallelization, but TPUs dominate at some point.


2. Jax-DEQN

AN implementation using Jax of the Deep Equilibrium Network (DEQN) in Jax.
The implantation allow the user to imput all the model logic in one simple class.
All the version can be run in CPU, GPU and TPUs. But, there are versions that are 
design to exploit the performance of GPUs and TPUs separately.
ALso, there are version with Batch Normalization, which are almos the same but the code 
is a bit harder to read.

3. Pre-train (or Fit) a Neural Net to a pre-specified policy.

Here I show how we can take an approximate solution
and use it to "pre-train" a neural net.
Our approach uses the first-order log linear approximation from dynare to pre-train the neural net.
The code is simple and it is optimized for GPU and TPUs.
THere are examples for a Real Business Cycle (Rbc) model and an Rbv with Production Network (RbcProdNet).




