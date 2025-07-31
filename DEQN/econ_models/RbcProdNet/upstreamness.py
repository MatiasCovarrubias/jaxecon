import jax.numpy as jnp


def calculate_upstreamness(model):
    """Calculate the upstreamness of each sector based on intermediate inputs and investment flows"""
    # Process policy
    policies_ss = jnp.exp(model.policies_ss)
    Pk = policies_ss[2 * model.n_sectors : 3 * model.n_sectors]
    Pm = policies_ss[3 * model.n_sectors : 4 * model.n_sectors]
    M = policies_ss[4 * model.n_sectors : 5 * model.n_sectors]
    Mout = policies_ss[5 * model.n_sectors : 6 * model.n_sectors]
    I = policies_ss[6 * model.n_sectors : 7 * model.n_sectors]
    P = policies_ss[8 * model.n_sectors : 9 * model.n_sectors]
    Q = policies_ss[9 * model.n_sectors : 10 * model.n_sectors]

    # Create identity matrix
    identity = jnp.eye(model.n_sectors)
    ones = jnp.ones(model.n_sectors)

    # Calculate Delta^M matrix (intermediate input upstreamness)
    P_term_M = jnp.outer(P ** (-model.sigma_m), Pm**model.sigma_m)
    M_Q_term = jnp.outer(1 / Q, M)
    Delta_M = model.Gamma_M * P_term_M * M_Q_term
    # Print row sums
    row_sums = jnp.sum(Delta_M, axis=1)
    print("Row sums of Delta_M:", row_sums)
    # Print column sums
    col_sums = jnp.sum(Delta_M, axis=0)
    print("Column sums of Delta_M:", col_sums)

    # Calculate Delta^I matrix (investment flow upstreamness)
    # Delta^I = Gamma_I * [(P^(-sigma_I)) * (Pk^sigma_I)] * [1_N * (I * Q^(-1))']
    P_term_I = jnp.outer(P ** (-model.sigma_I), Pk**model.sigma_I)
    I_Q_term = jnp.outer(1 / Q, I)
    Delta_I = model.Gamma_I * P_term_I * I_Q_term
    # Print row sums
    row_sums = jnp.sum(Delta_I, axis=1)
    print("Row sums of Delta_I:", row_sums)
    # Print column sums
    col_sums = jnp.sum(Delta_I, axis=0)
    print("Column sums of Delta_I:", col_sums)

    # Calculate upstreamness measures
    # U^M = [I - Delta^M]^(-1) * 1
    # U^I = [I - Delta^I]^(-1) * 1
    U_M = jnp.linalg.solve(identity - Delta_M, ones)
    U_I = jnp.linalg.solve(identity - Delta_I, ones)

    # Calculate alternative upstreamness measure: Mout/Q
    U_simple = Mout / Q

    # Create a dictionary with sector labels and upstreamness measures
    upstreamness_data = {"sectors": model.labels, "U_M": U_M, "U_I": U_I, "U_simple": U_simple}

    return upstreamness_data
