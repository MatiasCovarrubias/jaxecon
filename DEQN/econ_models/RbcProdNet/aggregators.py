import jax.numpy as jnp


def aggregator_fixedprices(model, simul_obs, simul_policy, Pvec, Pkvec, Pmvec):
    """Aggregate variables using fixed price vectors. Takes model, simul_obs (t,3*n), simul_policy (t,11*n+5), and price vectors (n,); returns dict of log deviations."""
    # Validate input dimensions
    if simul_obs.ndim != 2 or simul_policy.ndim != 2:
        raise ValueError("simul_obs and simul_policy must be 2D arrays")

    # Process obs and policy
    obs_notnorm = simul_obs * model.obs_sd + model.obs_ss  # denormalize
    policy_notnorm = simul_policy * model.policies_ss_level

    # Extract variables
    K = jnp.exp(obs_notnorm[:, : model.n_sectors])  # Shape: (T_simul, n_sectors)
    C = policy_notnorm[:, : model.n_sectors]
    M = policy_notnorm[:, 4 * model.n_sectors : 5 * model.n_sectors]
    I = policy_notnorm[:, 6 * model.n_sectors : 7 * model.n_sectors]
    Y = policy_notnorm[:, 10 * model.n_sectors : 11 * model.n_sectors]

    # Aggregate across sectors (axis=1) for each time period
    Kagg = jnp.sum(K * Pkvec, axis=1)  # Sum across sectors, result: (T_simul,)
    Cagg = jnp.sum(C * Pvec, axis=1)
    Magg = jnp.sum(M * Pmvec, axis=1)
    Iagg = jnp.sum(I * Pkvec, axis=1)
    Yagg = jnp.sum(Y * Pvec, axis=1)

    # Get deterministic steady state values
    K_detss = jnp.exp(model.obs_ss[: model.n_sectors])
    Kagg_detss = jnp.dot(K_detss, Pkvec)
    Kagg_devs = jnp.log(Kagg / Kagg_detss)

    C_detss = jnp.exp(model.policies_ss[: model.n_sectors])
    Cagg_detss = jnp.dot(C_detss, Pvec)
    Cagg_devs = jnp.log(Cagg / Cagg_detss)

    M_detss = jnp.exp(model.policies_ss[4 * model.n_sectors : 5 * model.n_sectors])
    Magg_detss = jnp.dot(M_detss, Pmvec)
    Magg_devs = jnp.log(Magg / Magg_detss)

    I_detss = jnp.exp(model.policies_ss[6 * model.n_sectors : 7 * model.n_sectors])
    Iagg_detss = jnp.dot(I_detss, Pkvec)
    Iagg_devs = jnp.log(Iagg / Iagg_detss)

    Y_detss = jnp.exp(model.policies_ss[10 * model.n_sectors : 11 * model.n_sectors])
    Yagg_detss = jnp.dot(Y_detss, Pvec)
    Yagg_devs = jnp.log(Yagg / Yagg_detss)

    return {
        "Kagg_fixedprices": Kagg_devs,
        "Cagg_fixedprices": Cagg_devs,
        "Magg_fixedprices": Magg_devs,
        "Iagg_fixedprices": Iagg_devs,
        "Yagg_fixedprices": Yagg_devs,
    }


def tornqvist_index_vectorized(quantities, prices):
    """
    Vectorized implementation of Tornqvist index.
    """
    # Calculate nominal values
    nominal_values = quantities * prices

    # Calculate value shares for each period
    total_values = jnp.sum(nominal_values, axis=1, keepdims=True)
    shares = nominal_values / total_values

    # Calculate log growth rates of quantities
    log_q_growth = jnp.log(quantities[1:, :] / quantities[:-1, :])

    # Calculate average shares between consecutive periods
    avg_shares = 0.5 * (shares[1:, :] + shares[:-1, :])

    # Calculate weighted log growth
    weighted_log_growth = jnp.sum(avg_shares * log_q_growth, axis=1)

    # Calculate cumulative index
    log_index = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(weighted_log_growth)])

    return log_index


def aggregator_tornqvist(
    model,
    simul_obs,
    simul_policy,
    base_period=0,
):
    """
    This function aggregate K, C, M, I, Y using a Tornqvist index.

    Input:
    model: Model object instance
    simul_obs has dimension (T_simul, 3*n_sectors).
    simul_policy has dimension (T_simul, 11*n_sectors+5).

    Process:
    Extract quantities and prices. Aggregate using Tornqvist formula.

    Output:
    Dictionary with new series for Kagg_tornqvist, Cagg_tornqvist, Magg_tornqvist, Iagg_tornqvist, Yagg_tornqvist.
    """

    obs_to_use = simul_obs
    policy_to_use = simul_policy

    # Process obs and policy - note: assuming time is first dimension
    obs_notnorm = obs_to_use * model.obs_sd + model.obs_ss  # denormalize
    policy_notnorm = policy_to_use * jnp.exp(model.policies_ss)

    # Extract variables (T_simul, n_sectors) or (T_simul+1, n_sectors) if steady state added
    K = jnp.exp(obs_notnorm[:, : model.n_sectors])
    C = policy_notnorm[:, : model.n_sectors]
    M = policy_notnorm[:, 4 * model.n_sectors : 5 * model.n_sectors]
    I = policy_notnorm[:, 6 * model.n_sectors : 7 * model.n_sectors]
    Y = policy_notnorm[:, 10 * model.n_sectors : 11 * model.n_sectors]

    # Extract prices
    P = policy_notnorm[:, 8 * model.n_sectors : 9 * model.n_sectors]
    Pk = policy_notnorm[:, 2 * model.n_sectors : 3 * model.n_sectors]
    Pm = policy_notnorm[:, 3 * model.n_sectors : 4 * model.n_sectors]

    # Calculate Tornqvist indices for each variable
    Kagg_tornqvist = tornqvist_index_vectorized(K, Pk)
    Cagg_tornqvist = tornqvist_index_vectorized(C, P)
    Magg_tornqvist = tornqvist_index_vectorized(M, Pm)
    Iagg_tornqvist = tornqvist_index_vectorized(I, Pk)
    Yagg_tornqvist = tornqvist_index_vectorized(Y, P)

    # Create dictionary with new series
    return {
        "Kagg_tornqvist": Kagg_tornqvist,
        "Cagg_tornqvist": Cagg_tornqvist,
        "Magg_tornqvist": Magg_tornqvist,
        "Iagg_tornqvist": Iagg_tornqvist,
        "Yagg_tornqvist": Yagg_tornqvist,
    }
