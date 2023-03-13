import numpy as np
from numpy import pi
from math import sqrt, log, exp, erf, erfc

from scipy.integrate import dblquad
from numba import njit
from robust_regression.aux_functions.likelihood_channel_functions import (
    Z_out_Bayes_decorrelated_noise,
)


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
    # print("first term ", first_term, "second term ", second_term, "third term ", third_term)
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
    # if m**2 <= q + q * delta_in:
    #     if m**2 * beta**2 >= q * (beta**2 + delta_out):
    #         return (1 + q + delta_in - percentage + 2*m*(-1 + percentage + beta*percentage) - percentage*(2*q + beta**2 + delta_in + delta_out))/(2.*(1 + sigma))
    #     else:
    #         return (1 + q + delta_in - delta_in*percentage - 2*m*(1 + (-1 + beta)*percentage) + percentage*(-1 + beta**2 + delta_out))/(2.*(1 + sigma))
    # else:
    #     if m**2 * beta**2 >= q * (beta**2 + delta_out):
    #         return -(1 + q + delta_in - delta_in*percentage - 2*m*(1 + (-1 + beta)*percentage) + percentage*(-1 + beta**2 + delta_out))/(2.*(1 + sigma))
    #     else:
    #         return -(1 + q + delta_in - percentage + 2*m*(-1 + percentage + beta*percentage) - percentage*(2*q + beta**2 + delta_in + delta_out))/(2.*(1 + sigma))


#     sigma = Q + q
#     # first_term = 1.0
#     # second_term = 1.0
#     return (
#         1
#         + q
#         + delta_in
#         - delta_in * percentage
#         - 2 * m * (1 + (-1 + beta) * percentage)
#         + percentage * (-1 + beta**2 + delta_out)
#     ) / (2 * (1 + sigma))
#     # return (first_term + second_term) / (2 * (1 + sigma))
#     # return (
#     #     1
#     #     + q
#     #     + delta_in
#     #     - delta_in * percentage
#     #     - 2 * m * (1 + (-1 + beta) * percentage)
#     #     + percentage * (-1 + beta**2 + delta_out)
#     # ) / (2 * (1 + sigma))


def integral_Psi_L2(y, xi, m, q, sigma, delta_in, delta_out, percentage, beta):
    return (
        np.exp(-0.5 * xi**2)
        / np.sqrt(2 * np.pi)
        * (y - np.sqrt(q) * xi) ** 2
        / (2 * (1 + sigma))
        * Z_out_Bayes_decorrelated_noise(
            y, m * xi / np.sqrt(q), 1 - m**2 / q, delta_in, delta_out, percentage, beta
        )
    )


def Psi_out_L2_num(Q, m, q, delta_in, delta_out, percentage, beta):
    int_result = dblquad(
        integral_Psi_L2,
        -np.inf,
        np.inf,
        -np.inf,
        np.inf,
        args=(m, q, Q - q, delta_in, delta_out, percentage, beta),
    )

    return int_result[0]


@njit
def Zout_L1(y, omega, sigma):
    return
    # return np.exp(-0.5 * ((y - omega) ** 2) / sigma) / (np.pi * np.sqrt(sigma))


@njit
def integral_Psi_L1(y, xi, m, q, sigma, delta_in, delta_out, percentage, beta):
    return (
        (np.exp(-0.5 * xi**2) / np.sqrt(2 * np.pi))
        * Z_out_Bayes_decorrelated_noise(
            y, m * xi / np.sqrt(q), 1 - m**2 / q, delta_in, delta_out, percentage, beta
        )
        * np.log(Zout_L1(y, np.sqrt(q) * xi, sigma))
    )


def Psi_out_L1(Q, m, q, delta_in, delta_out, percentage, beta):
    sigma = Q - q
    return (
        (
            sqrt(2 * pi)
            * (2 + sigma)
            * (
                exp(
                    (
                        sigma**2
                        * (
                            -(1 / (1 - 2 * m + q + delta_in))
                            + 1 / (q - 2 * m * beta + beta**2 + delta_out)
                        )
                    )
                    / 2.0
                )
                * (-1 + 2 * m - q - delta_in)
                * (-1 + percentage)
                + sqrt(1 - 2 * m + q + delta_in)
                * percentage
                * sqrt(q - 2 * m * beta + beta**2 + delta_out)
            )
        )
        / (
            exp(sigma**2 / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
            * sqrt(1 - 2 * m + q + delta_in)
        )
        + pi
        * (
            1
            + q
            + delta_in
            + 2 * m * (-1 + percentage)
            - (1 + q + sigma + sigma**2 + delta_in) * percentage
        )
        * erf(sigma / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
        + pi
        * percentage
        * (q + sigma + sigma**2 - 2 * m * beta + beta**2 + delta_out)
        * erf(sigma / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out)))
        - pi * sigma * (1 + sigma) * erfc(sigma / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
    ) / (2.0 * pi * (1 + sigma))
    # sigma = Q - q
    # first_piece = -(
    #     (
    #         percentage
    #         * (
    #             -(m * q)
    #             - m**2 * (-1 + beta**2)
    #             + q * (q + beta**2 + delta_out)
    #             + m * q * np.sign(m**2 * beta**2 - q * (beta**2 + delta_out))
    #         )
    #     )
    #     / (np.pi * q * np.sqrt(sigma))
    # )

    # second_piece = ((1 - 2 * m + q + delta_in) * (-1 + percentage)) / (np.pi * np.sqrt(sigma))

    # third_piece = -np.log(np.pi * np.sqrt(sigma))

    # return first_piece + second_piece + third_piece


def Psi_out_L1_num(Q, m, q, delta_in, delta_out, percentage, beta):
    int_result = dblquad(
        integral_Psi_L1,
        -np.inf,
        np.inf,
        -np.inf,
        np.inf,
        args=(m, q, Q - q, delta_in, delta_out, percentage, beta),
    )[0]

    return int_result


def Psi_out_Huber(Q, m, q, delta_in, delta_out, percentage, beta, a):
    sigma = Q - q
    return (
        (
            sqrt(1 / (1 - 2 * m + q + delta_in))
            * (1 - percentage)
            * (
                -2
                * (1 + 2 * a)
                * exp((1 + sigma) ** 2 / (4 * m - 2 * (1 + q + delta_in)))
                * q
                * (1 - 2 * m + q + delta_in)
                + (
                    sqrt(2 * pi)
                    * q
                    * sqrt(1 - 2 * m + q + delta_in)
                    * (
                        (1 - 2 * m + q + delta_in)
                        * erf((1 + sigma) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
                        + a**2
                        * (1 + sigma) ** 2
                        * erfc((1 + sigma) / (sqrt(2) * sqrt(1 - 2 * m + q + delta_in)))
                    )
                )
                / (1 + sigma)
            )
        )
        / q
        + (
            percentage
            * (
                (-2 * (1 + 2 * a) * q * (q - 2 * m * beta + beta**2 + delta_out))
                / exp((1 + sigma) ** 2 / (2.0 * (q - 2 * m * beta + beta**2 + delta_out)))
                + (
                    sqrt(2 * pi)
                    * q
                    * sqrt(q - 2 * m * beta + beta**2 + delta_out)
                    * (
                        (q - 2 * m * beta + beta**2 + delta_out)
                        * erf(
                            (1 + sigma) / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out))
                        )
                        + a**2
                        * (1 + sigma) ** 2
                        * erfc(
                            (1 + sigma) / (sqrt(2) * sqrt(q - 2 * m * beta + beta**2 + delta_out))
                        )
                    )
                )
                / (1 + sigma)
            )
        )
        / sqrt(q**2 * (q - 2 * m * beta + beta**2 + delta_out))
    ) / (2.0 * sqrt(2 * pi))
