from math import exp, sqrt
from numpy import pi
from numba import vectorize, njit


@vectorize('float64(float64, float64, float64)')
def std_gaussian(x: float, mu: float, sigma_2: float) -> float:
    return exp(-0.5 * pow(x - mu, 2.0) / sigma_2) / sqrt(2 * pi * sigma_2)


@vectorize
def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old


@njit(error_model="numpy", fastmath=True)
def gen_error(m, q, sigma, *args):
    return 1 + q - 2 * m