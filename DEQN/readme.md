This folder contains a custom implementation of the [DEQN (Deep Equilibrium Network)](https://onlinelibrary.wiley.com/doi/full/10.1111/iere.12575?msockid=13945c9feebc6c902c704988efca6d9f) algorithm written in jax. The easiest way to run the algorithm is to use Google Colab, which gives you access to GPUs. To see an example of how to run it, open the Rbc_CES.ipynb notebook, and run it in google colab. To see an example that shows and explain all the functions of the algorithms, open the notebook jaxDEQN.ipynb and run it in google colab.

The implementation of the algorithm has the following subfolders

-   econ_models: This subfolder contains the economic models implemented so far. Economic models are implemented as python classes. Each economic model has its own .py document, but that file may contain many classes corresponding to different versions of the model.
-   algorithm: This subfolder contains the files that make up the algorithm. Each file contains a type of function, of which several versions can coexist in the same file. The types of functions are: get_simulation_fn(), get_loss_fn(), get_epoch_update_fn and get_eval_fn.
-   experiments: This subfolder contains a set of experiments that use different econ models and different components of the algorithm. Look here if you want to look at examples, with all their configs and everything. Experiments are implemented using colabs notebooks (.ipynbs).
-   analysis: This subfolder contains the scripts of all the analysis we can make on a candidate solution.
-   neural_nets: This folder contains .py files for neural nets.
-   tests: This subfolder contains all the test scripts.
