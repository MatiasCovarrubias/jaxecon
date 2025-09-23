#!/usr/bin/env python3
"""
RBC CES Model Training Script

This script trains a neural network to output the optimal policy of an RBC model
with CES production function using the DEQN solver.

Converted from Rbc_CES.ipynb for local development.
"""

from time import time

import jax
import jaxopt
import matplotlib.pyplot as plt
import optax
from algorithm.epoch_train import create_epoch_train_fn
from algorithm.eval import create_eval_fn
from algorithm.loss import create_batch_loss_fn
from algorithm.simulation import create_episode_simul_fn
from analysis.simul_analysis import create_descstats_fn, create_episode_simul_verbose_fn
from analysis.stochastic_ss import create_stochss_fn
from econ_models.rbc_ces import RbcCES, RbcCES_SteadyState
from flax.training.train_state import TrainState
from jax import config as jax_config
from jax import numpy as jnp
from jax import random

# Local imports
from neural_nets.neural_nets import NeuralNet


def setup_jax(double_precision=True):
    """Configure JAX settings."""
    jax_config.update("jax_debug_nans", True)

    if double_precision:
        jax_config.update("jax_enable_x64", True)
        precision = jnp.float64
    else:
        precision = jnp.float32

    return precision


def create_econ_model(precision):
    """Create and configure the economic model."""
    # Set parameters
    beta = 0.96
    alpha = 0.31
    delta = 0.057
    sigma_y = 0.5
    eps_c = 0.5
    eps_l = 0.5
    rho = 0.69
    phi = 5
    shock_sd = 0.0153

    # Find steady state and theta
    econ_model_ss = RbcCES_SteadyState(
        precision=precision, beta=beta, alpha=alpha, delta=delta, sigma_y=sigma_y, eps_c=eps_c, eps_l=eps_l
    )

    initial_policy = jnp.array(
        [1.0000126, 0.41989392, 2.2284806, 0.12702352, 0.99998146, 0.99996126, 1.1270372, 12.25385]
    )

    @jax.jit
    def optimize_policy(initial_policy):
        solver = jaxopt.BFGS(econ_model_ss.loss, tol=1e-09, verbose=False)
        ss_policy, state = solver.run(initial_policy)
        return ss_policy, state

    ss_policy, state = optimize_policy(initial_policy)
    print("Steady state optimization loss:", state.error)
    print("Steady state solution:", ss_policy)

    policies_ss = jnp.log(ss_policy[:7])
    theta = ss_policy[-1]
    print("Theta:", theta)

    # Create economic model
    econ_model = RbcCES(
        precision=precision,
        policies_ss=policies_ss,
        theta=theta,
        beta=beta,
        alpha=alpha,
        delta=delta,
        sigma_y=sigma_y,
        eps_c=eps_c,
        eps_l=eps_l,
        rho=rho,
        phi=phi,
        shock_sd=shock_sd,
    )

    return econ_model


def create_config():
    """Create experiment configuration."""
    config = {
        # general
        "seed": 1,
        "exper_name": "RbcCES_local_run",
        # TODO: Implement save directory for local runs
        "save_dir": "./results/",  # Will be implemented later
        "restore": False,
        "restore_exper_name": "",
        # neural net
        "layers": [32, 32],
        # learning rate schedule
        "lr_sch_values": [0.001, 0.001],
        "lr_sch_transitions": [900],
        "lr_end_value": 1e-7,
        # simulation
        "periods_per_epis": 64,
        "simul_vol_scale": 1,
        "init_range": 2,
        # loss calculation
        "mc_draws": 8,
        # training
        "epis_per_step": 64,
        "steps_per_epoch": 100,
        "n_epochs": 10,
        "batch_size": 16,
        
        "checkpoint_frequency": 1000,
        "config_eval": {
            "periods_per_epis": 64,
            "mc_draws": 256,
            "simul_vol_scale": 1,
            "eval_n_epis": 128,
            "init_range": 0,
        },
    }

    # Create auxiliary config variables for readability
    config["periods_per_step"] = config["periods_per_epis"] * config["epis_per_step"]
    config["n_batches"] = config["periods_per_step"] // config["batch_size"]

    return config


def print_model_info(config, precision):
    """Print model and experiment information."""
    print("Number of parameters:")
    nn_info = NeuralNet(config["layers"] + [RbcCES().n_actions], precision)
    print(nn_info.tabulate(random.PRNGKey(0), RbcCES(precision=precision).initial_obs(random.PRNGKey(0))))

    total_steps = config["steps_per_epoch"] * config["n_epochs"]
    print(f"TOTAL Number of steps (NN updates): {total_steps} steps\n")


def run_experiment(econ_model, config, precision):
    """Run the training experiment."""
    print("Starting experiment...")

    # CREATE NN, RNGS, TRAIN_STATE AND EPOCH UPDATE
    nn = NeuralNet(features=config["layers"] + [econ_model.n_actions], precision=precision)
    rng, rng_pol, rng_econ_model, rng_epoch, rng_eval = random.split(random.PRNGKey(config["seed"]), num=5)

    # CREATE LR SCHEDULE
    lr_schedule = optax.join_schedules(
        schedules=[optax.constant_schedule(i) for i in config["lr_sch_values"][:-1]]
        + [
            optax.warmup_cosine_decay_schedule(
                init_value=config["lr_sch_values"][-1],
                peak_value=config["lr_sch_values"][-1],
                warmup_steps=0,
                decay_steps=config["n_epochs"] * config["steps_per_epoch"] - config["lr_sch_transitions"][-1],
                end_value=config["lr_end_value"],
            )
        ],
        boundaries=config["lr_sch_transitions"],
    )

    # INITIALIZE FULL NN TRAIN STATE
    if not config["restore"]:
        params = nn.init(rng_pol, jnp.zeros_like(econ_model.initial_obs(rng_econ_model)))
        train_state = TrainState.create(apply_fn=nn.apply, params=params, tx=optax.adam(lr_schedule))
    else:
        # TODO: Implement checkpoint restoration for local runs
        raise NotImplementedError("Checkpoint restoration not yet implemented for local runs")

    # GET TRAIN AND EVAL FUNCTIONS
    simul_fn = jax.jit(create_episode_simul_fn(econ_model, config))
    loss_fn = jax.jit(create_batch_loss_fn(econ_model, config))
    train_epoch_fn = jax.jit(create_epoch_train_fn(econ_model, config, simul_fn, loss_fn))
    eval_fn = jax.jit(create_eval_fn(config, simul_fn, loss_fn))

    # COMPILE CODE
    print("Compiling functions...")
    time_start = time()
    train_epoch_fn(train_state, rng_epoch)  # compiles
    eval_fn(train_state, rng_epoch)  # compiles
    time_compilation = time() - time_start
    print(f"Time Elapsed for Compilation: {time_compilation:.2f} seconds")

    # RUN AN EPOCH TO GET TIME STATS
    time_start = time()
    train_epoch_fn(train_state, rng_epoch)  # run one epoch
    time_epoch = time() - time_start
    print(f"Time Elapsed for epoch: {time_epoch:.2f} seconds")

    time_start = time()
    eval_fn(train_state, rng_epoch)  # run one epoch
    time_eval = time() - time_start
    print(f"Time Elapsed for eval: {time_eval:.2f} seconds")

    time_experiment = (time_epoch + time_eval) * config["n_epochs"] / 60
    print(f"Estimated time for full experiment: {time_experiment:.2f} minutes")

    steps_per_second = config["steps_per_epoch"] * config["periods_per_step"] / time_epoch
    print(f"Steps per second: {steps_per_second:.0f} st/s")

    # CREATE LISTS TO STORE METRICS
    mean_losses, max_losses, mean_accuracy, min_accuracy = [], [], [], []

    # RUN ALL THE EPOCHS
    print("\nStarting training...")
    time_start = time()
    for i in range(1, config["n_epochs"] + 1):
        # eval
        eval_metrics = eval_fn(train_state, rng_eval)
        print("EVALUATION:")
        print(f"  Iteration: {train_state.step}")
        print(f"  Mean Loss: {eval_metrics[0]:.6e}")
        print(f"  Max Loss: {eval_metrics[1]:.6e}")
        print(f"  Mean Acc: {eval_metrics[2]:.6f}")
        print(f"  Min Acc: {eval_metrics[3]:.6f}")
        print(f"  Mean Accs Foc: {eval_metrics[4]}")
        print(f"  Min Accs Foc: {eval_metrics[5]}")

        # run epoch
        train_state, rng_epoch, epoch_metrics = train_epoch_fn(train_state, rng_epoch)
        print("TRAINING:")
        print(f"  Iteration: {train_state.step}")
        print(f"  Mean Loss: {jnp.mean(epoch_metrics[0]):.6e}")
        print(f"  Max Loss: {jnp.mean(epoch_metrics[1]):.6e}")
        print(f"  Mean Acc: {jnp.mean(epoch_metrics[2]):.6f}")
        print(f"  Min Acc: {jnp.min(epoch_metrics[3]):.6f}")
        print(f"  Learning rate: {lr_schedule(train_state.step):.6e}")
        print()

        # TODO: Implement checkpointing for local runs
        if (
            train_state.step >= config["checkpoint_frequency"]
            and train_state.step % config["checkpoint_frequency"] == 0
        ):
            # checkpoints.save_checkpoint(ckpt_dir=config['save_dir']+config['exper_name'], target=train_state, step=train_state.step)

            # store results
            mean_losses.append(float(eval_metrics[0]))
            max_losses.append(float(eval_metrics[1]))
            mean_accuracy.append(float(eval_metrics[2]))
            min_accuracy.append(float(eval_metrics[3]))

    # PRINT RESULTS
    print("=" * 50)
    print("FINAL RESULTS:")
    print(f"Minimum mean loss attained in evaluation: {min(mean_losses):.6e}")
    print(f"Minimum max loss attained in evaluation: {min(max_losses):.6e}")
    print(f"Maximum mean accuracy attained in evaluation: {max(mean_accuracy):.6f}")
    print(f"Maximum min accuracy attained in evaluation: {max(min_accuracy):.6f}")
    time_fullexp = (time() - time_start) / 60
    print(f"Time Elapsed for Full Experiment: {time_fullexp:.2f} minutes")

    # TODO: Implement results storage for local runs
    results = {
        "exper_name": config["exper_name"],
        "min_mean_loss": min(mean_losses),
        "min_max_loss": min(max_losses),
        "max_mean_acc": max(mean_accuracy),
        "max_min_acc": max(min_accuracy),
        "time_full_exp_minutes": time_fullexp,
        "time_epoch_seconds": time_epoch,
        "time_compilation_seconds": time_compilation,
        "steps_per_second": steps_per_second,
        "config": config,
        "mean_losses_list": mean_losses,
        "max_losses_list": max_losses,
        "mean_acc_list": mean_accuracy,
        "min_acc_list": min_accuracy,
    }

    # TODO: Save results to JSON file
    # os.makedirs(config['save_dir'] + config['exper_name'], exist_ok=True)
    # with open(config['save_dir'] + config['exper_name'] + "/results.json", "w") as write_file:
    #     json.dump(results, write_file)

    # TODO: Generate and save plots
    # plot_learning_curves(config, mean_losses, max_losses, mean_accuracy, min_accuracy, lr_schedule)

    return train_state, results


def run_analysis(econ_model, trained_train_state):
    """Run analysis on the trained model."""
    print("\nRunning analysis...")

    config_analysis = {
        "init_range": 0,
        "periods_per_epis": 500,
        "simul_vol_scale": 1,
    }

    config_stochss = {"n_draws": 500, "time_to_converge": 200, "seed": 0}

    rng_analysis = random.PRNGKey(4)

    # create functions
    simul_fn_verbose = jax.jit(create_episode_simul_verbose_fn(econ_model, config_analysis))
    descstats_fn = create_descstats_fn(econ_model, config_analysis)
    stochss_fn = jax.jit(create_stochss_fn(econ_model, config_stochss))

    # simulate model
    simul_obs, simul_policies = simul_fn_verbose(trained_train_state, rng_analysis)
    simul_policies_logdev = jnp.log(simul_policies)
    descstats_df, autocorr_df = descstats_fn(simul_policies_logdev, simul_obs)
    print("\nDescriptive Statistics:")
    print(descstats_df)

    # calculate stochastic steady state
    policy_stochss, obs_stochss = stochss_fn(simul_obs, trained_train_state)
    print("\nStochastic Steady State Policy:")
    print(policy_stochss)
    print("\nStochastic Steady State Obs:")
    print(obs_stochss)

    return descstats_df, autocorr_df, (policy_stochss, obs_stochss)


def plot_learning_curves(config, mean_losses, max_losses, mean_accuracy, min_accuracy, lr_schedule):
    """Generate and save learning curve plots."""
    # TODO: Implement plotting and saving for local runs
    checkpoint_steps = [(i + 1) * config["checkpoint_frequency"] for i in range(len(mean_losses))]

    # Mean Losses
    plt.figure()
    plt.plot(checkpoint_steps, mean_losses)
    plt.xlabel("Steps (NN updates)")
    plt.ylabel("Mean Losses")
    plt.title("Training Progress: Mean Losses")
    # plt.savefig(config['save_dir']+config['exper_name']+'/mean_losses.jpg')
    plt.show()
    plt.close()

    # Max Losses
    plt.figure()
    plt.plot(checkpoint_steps, max_losses)
    plt.xlabel("Steps (NN updates)")
    plt.ylabel("Max Losses")
    plt.title("Training Progress: Max Losses")
    # plt.savefig(config['save_dir']+config['exper_name']+'/max_losses.jpg')
    plt.show()
    plt.close()

    # Mean Accuracy
    plt.figure()
    plt.plot(checkpoint_steps, mean_accuracy)
    plt.xlabel("Steps (NN updates)")
    plt.ylabel("Mean Accuracy (%)")
    plt.title("Training Progress: Mean Accuracy")
    # plt.savefig(config['save_dir']+config['exper_name']+'/mean_accuracy.jpg')
    plt.show()
    plt.close()

    # Min Accuracy
    plt.figure()
    plt.plot(checkpoint_steps, min_accuracy)
    plt.xlabel("Steps (NN updates)")
    plt.ylabel("Minimum Accuracy (%)")
    plt.title("Training Progress: Minimum Accuracy")
    # plt.savefig(config['save_dir']+config['exper_name']+'/min_accuracy.jpg')
    plt.show()
    plt.close()

    # Learning rate schedule
    plt.figure()
    plt.plot(checkpoint_steps, [lr_schedule(step) for step in checkpoint_steps])
    plt.xlabel("Steps (NN updates)")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    # plt.savefig(config['save_dir']+config['exper_name']+'/learning_rate.jpg')
    plt.show()
    plt.close()


def main():
    """Main function to run the complete experiment."""
    print("RBC CES Model Training")
    print("=" * 50)

    # Setup
    precision = setup_jax(double_precision=True)
    config = create_config()

    # Create economic model
    print("Creating economic model...")
    econ_model = create_econ_model(precision)

    # Print model information
    print_model_info(config, precision)

    # Run experiment
    trained_train_state, results = run_experiment(econ_model, config, precision)

    # Run analysis
    descstats_df, autocorr_df, stochss = run_analysis(econ_model, trained_train_state)

    print("\nExperiment completed successfully!")

    return trained_train_state, results, descstats_df, autocorr_df, stochss


if __name__ == "__main__":
    main()
