from numpy import pi
from math import sqrt, exp, erf, erfc
from numba import njit


@njit
def free_energy(
    Psi_w, Psi_out, alpha, m, q, sigma, m_hat, q_hat, sigma_hat, Psi_w_args=(), Psi_out_args=()
):
    Q_hat = sigma_hat - q_hat
    Q = sigma + q
    first_term = (
        -0.5 * (q * sigma_hat - q_hat * sigma) + m * m_hat
    )  # m * m_hat - 0.5 * q * q_hat - 0.5 * Q * Q_hat
    second_term = -Psi_w(Q_hat, m_hat, q_hat, *Psi_w_args)
    third_term = alpha * Psi_out(Q, m, q, *Psi_out_args)
    return first_term + second_term + third_term


@njit
def Psi_w_l2_reg(Q_hat, m_hat, q_hat, reg_param):
    reg_param_combination = Q_hat + q_hat + reg_param
    return 0.5 * ((q_hat + m_hat**2) / reg_param_combination)
    # return 0.5 * ((q_hat + m_hat**2) / reg_param_combination - log(reg_param_combination))


@njit
def Psi_out_L2(Q, m, q, delta_in, delta_out, percentage, beta):
    sigma = Q - q
    return (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (2.0 * (1 + sigma))


@njit
def Psi_out_L1(Q, m, q, delta_in, delta_out, percentage, beta):
    sigma = Q - q
    comb_in = 1 - 2 * m + q + delta_in
    comb_out = q - 2 * m * beta + beta**2 + delta_out
    return (
        (
            sqrt(2 * pi)
            * (2 + sigma)
            * (
                exp((sigma**2 * (-(1 / (comb_in)) + 1 / (comb_out))) / 2.0)
                * (-1 + 2 * m - q - delta_in)
                * (-1 + percentage)
                + sqrt(comb_in) * percentage * sqrt(comb_out)
            )
        )
        / (exp(sigma**2 / (2.0 * (comb_out))) * sqrt(comb_in))
        + pi
        * (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + sigma + sigma**2 + delta_in) * percentage
        )
        * erf(sigma / (sqrt(2) * sqrt(comb_in)))
        + pi
        * percentage
        * (q + sigma + sigma**2 - 2 * m * beta + beta**2 + delta_out)
        * erf(sigma / (sqrt(2) * sqrt(comb_out)))
        - pi * sigma * (1 + sigma) * erfc(sigma / (sqrt(2) * sqrt(comb_in)))
    ) / (2.0 * pi * (1 + sigma))


@njit
def Psi_out_Huber(Q, m, q, delta_in, delta_out, percentage, beta, a):
    sigma = Q - q
    comb_in = 1 - 2 * m + q + delta_in
    comb_out = q - 2 * m * beta + beta**2 + delta_out
    return (
        -(
            (
                a
                * sqrt(2 / pi)
                * (1 + sigma)
                * (
                    exp((a**2 * (1 + sigma) ** 2 * (-(1 / (comb_in)) + 1 / (comb_out))) / 2.0)
                    * sqrt(comb_in)
                    * (-1 + percentage)
                    - percentage * sqrt(comb_out)
                )
            )
            / exp((a**2 * (1 + sigma) ** 2) / (2.0 * (comb_out)))
        )
        + (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + a**2 * (1 + sigma) ** 2 + delta_in) * percentage
        )
        * erf((a * (1 + sigma)) / (sqrt(2) * sqrt(comb_in)))
        + percentage
        * (q + a**2 * (1 + sigma) ** 2 - 2 * m * beta + beta**2 + delta_out)
        * erf((a * (1 + sigma)) / (sqrt(2) * sqrt(comb_out)))
        - a**2 * (1 + sigma) ** 2 * erfc((a * (1 + sigma)) / (sqrt(2) * sqrt(comb_in)))
    ) / (2.0 * (1 + sigma))
