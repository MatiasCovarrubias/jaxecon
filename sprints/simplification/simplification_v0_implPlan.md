# The objective of this sprint is to simplify parts of the DEQN algorithm, since we have learned that many tricks were not needed.

We need to simplify:

-   Simulation function now dont need to have proxied versions. We want to keep that simulation function as an advance version, but not the main one.
-   We dont need to have state augmentation. Thus, in the model, there is no need to distinguish between observation and state.
-   In the models, before we had a different normalization for state vs policy. Now, both are expressed as log deviations from deterministic steady state, scaled by the standard deviation of states (calculated previously with a first order log approximation of the model).
-   Thus, the neural net dont need softplus at the end. We map log deviations to log deviations. But, we now have a version that accepts a loglinear policy matrix that can be used as a baseline, so the neural nets learns the ressidual.
