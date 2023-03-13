import numpy as np
from math import sqrt, log

from scipy.integrate import dblquad
from numba import njit

@njit
def training_error_ridge(m, q, sigma, reg_param, alpha, delta_in, delta_out, percentage, beta):
    return reg_param / (2 * alpha) * q + (
        1
        + q
        + delta_in
        - delta_in * percentage
        - 2 * m * (1 + (-1 + beta) * percentage)
        + percentage * (-1 + beta**2 + delta_out)
    ) / (2 * (1 + sigma) ** 2)


def training_error_l1(m, q, sigma, reg_param):
    raise NotImplementedError


def training_error_huber(m, q, sigma, reg_param):
    raise NotImplementedError
