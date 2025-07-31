This folder contains a custom implementation of the DEQN (Deep Equilibrium Network) algorithm written in jax. The algorithm is designed to be authomatic to use with Google Colab GPUs. To see an example of how to run it, open the Rbc_CES.ipynb notebook, and run it in google colab. To see an example that shows and explain all the functions of the algorithms, open the notebook jaxDEQN.ipynb and run it in google colab.

THe implementation of the algorithm has the following subfolders
- econ_models: This subfolder contain all the economic models implemented. Economic models are implemented as class. Each economic model has its own .py document, but that file may contain many classes corresponding to different versions of the model. 
- algorithm: This subfolder contain the files that make up the algorithm. Each file contains a type of function, of which several versions can coesxist in the same file. The types of functions are: get_simulation_fn(), get_loss_fn(), get_epoch_update_fn and get_eval_fn.
- experiments: This subfolder contain a set of experiments that use different econ models and different components of the algorithm. Look here if you want to look at examples, with all their configs and everything. Experiments are implemented using colabs notebooks (.ipynbs).
- analysis: This subfolder contains the scripts of all the analysis we can make on checkpointed solution.
- neural_nets: This folder contain .py files for each type of neural net. For example, multi layer perceptrons have a file, but there are different version.
- tests: This subfolder contain all the test scripts.
