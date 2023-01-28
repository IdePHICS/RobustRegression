from numba import njit
from numpy import pi
from math import erf, erfc, exp, log, sqrt


@njit(error_model="numpy", fastmath=True)
def var_func_L2(m_hat, q_hat, sigma_hat, reg_param):
    m = m_hat / (sigma_hat + reg_param)
    q = (m_hat**2 + q_hat) / (sigma_hat + reg_param) ** 2
    sigma = 1.0 / (sigma_hat + reg_param)
    return m, q, sigma


@njit(error_model="numpy", fastmath=True)
def var_hat_func_Huber_single_noise(m, q, sigma, alpha, delta, a):
    arg_sqrt = 1 + q + delta - 2 * m
    erf_arg = (a * (sigma + 1)) / sqrt(2 * arg_sqrt)

    m_hat = (alpha / (1 + sigma)) * erf(erf_arg)
    q_hat = (alpha / (1 + sigma) ** 2) * (
        arg_sqrt * erf(erf_arg)
        + a**2 * (1 + sigma) ** 2 * erfc(erf_arg)
        - a * (1 + sigma) * sqrt(2 / pi) * sqrt(arg_sqrt) * exp(-(erf_arg**2))
    )
    sigma_hat = (alpha / (1 + sigma)) * erf(erf_arg)
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_Huber_double_noise(m, q, sigma, alpha, delta_in, delta_out, percentage, a):
    small_sqrt = delta_in - 2 * m + q + 1
    large_sqrt = delta_out - 2 * m + q + 1
    small_erf = (a * (sigma + 1)) / sqrt(2 * small_sqrt)
    large_erf = (a * (sigma + 1)) / sqrt(2 * large_sqrt)

    m_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    q_hat = alpha * (
        a**2
        - (sqrt(2 / pi) * a / (1 + sigma))
        * (
            (1 - percentage) * sqrt(small_sqrt) * exp(-(small_erf**2))
            + percentage * sqrt(large_sqrt) * exp(-(large_erf**2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage) * (small_sqrt - (a * (1 + sigma)) ** 2) * erf(small_erf)
            + percentage * (large_sqrt - (a * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigma_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    return m_hat, q_hat, sigma_hat


@njit(error_model="numpy", fastmath=True)
def var_hat_func_Huber_decorrelated_noise(
    m, q, sigma, alpha, delta_in, delta_out, percentage, beta, a
):
    small_sqrt = delta_in - 2 * m + q + 1
    large_sqrt = delta_out - 2 * m * beta + q + beta**2
    small_erf = (a * (sigma + 1)) / sqrt(2 * small_sqrt)
    large_erf = (a * (sigma + 1)) / sqrt(2 * large_sqrt)

    m_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
    )
    q_hat = alpha * (
        a**2
        - (sqrt(2 / pi) * a / (1 + sigma))
        * (
            (1 - percentage) * sqrt(small_sqrt) * exp(-(small_erf**2))
            + percentage * sqrt(large_sqrt) * exp(-(large_erf**2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage) * (small_sqrt - (a * (1 + sigma)) ** 2) * erf(small_erf)
            + percentage * (large_sqrt - (a * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigma_hat = (alpha / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )
    return m_hat, q_hat, sigma_hat
