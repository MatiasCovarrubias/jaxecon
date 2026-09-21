"""Attach a computed log-linear policy as the APG network baseline."""

from jax import numpy as jnp

from APG.neural_nets import ActorCritic, PolicyNet, PolicyNetLoglinear

from .solve import print_loglinear_solution, solve_loglinear


def prepare_loglinear(env, config, rng=None):
    """Solve the LQ policy before training when ``config["loglinear_baseline"]`` is set.

    Keys:
        loglinear_baseline: use ``C`` as the network baseline (default False)
        loglinear_normalize: set the environment ``state_sd`` / ``policies_sd``
            to the standard deviations of the linear solution (default True)
        loglinear_solver: ``"scipy"`` (default) or ``"iterate"``

    The solved ``C``, scales, and closed-loop spectral radius are written back
    into ``config`` so that they are saved with the run.
    """
    if not config.get("loglinear_baseline", False):
        return None
    solution = solve_loglinear(env, rng=rng, solver=config.get("loglinear_solver", "scipy"))
    print_loglinear_solution(solution)
    if config.get("loglinear_normalize", True):
        env.set_scales(solution.states_sd, solution.policies_sd)
    config["loglinear_C"] = solution.C.tolist()
    config["loglinear_states_sd"] = solution.states_sd.tolist()
    config["loglinear_policies_sd"] = solution.policies_sd.tolist()
    config["loglinear_spectral_radius"] = solution.spectral_radius
    return solution


def build_actor(env, config, precision, solution=None):
    """PolicyNet or ActorCritic; residual around ``solution.C`` when it is given.

    The baseline reads the environment's current scales, so it is consistent
    with whatever normalization ``prepare_loglinear`` applied.
    """
    layers = config.get("layers", config.get("layers_actor", [32, 16]))
    output_bias_init = config.get("policy_output_bias_init")
    loglinear = {}
    if solution is not None:
        loglinear = {
            "C": jnp.asarray(solution.C, dtype=precision),
            "states_sd": jnp.asarray(env.state_sd, dtype=precision),
            "policies_sd": jnp.asarray(env.policies_sd, dtype=precision),
        }

    if config.get("use_terminal_value", False):
        return ActorCritic(
            actions_dim=env.action_dim,
            hidden_dims_actor=layers,
            hidden_dims_critic=config.get("layers_critic", layers),
            precision=precision,
            policy_output_bias_init=output_bias_init,
            **{f"loglinear_{key}": value for key, value in loglinear.items()},
        )
    if loglinear:
        return PolicyNetLoglinear(features=layers, n_out=env.action_dim, precision=precision, **loglinear)
    return PolicyNet(
        features=layers,
        n_out=env.action_dim,
        precision=precision,
        output_bias_init=output_bias_init,
    )
