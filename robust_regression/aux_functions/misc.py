from math import exp, sqrt, acos
from numpy import pi, arccos
import numpy as np
from numba import vectorize, njit


@vectorize("float64(float64, float64, float64)")
def std_gaussian(x: float, mu: float, sigma_2: float) -> float:
    return exp(-0.5 * pow(x - mu, 2.0) / sigma_2) / sqrt(2 * pi * sigma_2)


@vectorize("float64(float64, float64, float64)")
def damped_update(new, old, damping):
    """
    Damped update of old value with new value.
    the opertation that is performed is:
    damping * new + (1 - damping) * old
    """
    return damping * new + (1 - damping) * old


# @njit(error_model="numpy", fastmath=True)
def gen_error(m, q, sigma, *args):
    return 1 + q - 2.0 * m


@njit
def angle_teacher_student(m, q, sigma, *args):
    return np.arccos(m / np.sqrt(q)) / pi


def gen_error_ML(m, q, sigma, delta_in, delta_out, percentage, beta):
    return 0.5*(
        q
        # + delta_in
        - 2 * m * (1 + (-1 + beta) * percentage)
        + 1
        # + percentage * (-delta_in + delta_out)
        + percentage * (-1 + beta**2)
    )


def gen_error_ML_BO(m, q, sigma, delta_in, delta_out, percentage, beta):
    q = (1-percentage + percentage * beta )**2 * q
    m = (1-percentage + percentage * beta ) * m

    return 0.5*(
        q
        # + delta_in
        - 2 * m * (1 + (-1 + beta) * percentage)
        + 1
        # + percentage * (-delta_in + delta_out)
        + percentage * (-1 + beta**2)
    )