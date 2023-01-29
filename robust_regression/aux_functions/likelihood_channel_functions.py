from numba import njit, vectorize
from numpy import pi, sign
from math import exp, sqrt, pow, abs

# the function *_Bayes could be made more safe by taking the largest one
@njit(error_model="numpy", fastmath=True)
def Z_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_small: float,
    delta_large: float,
    eps: float,
    beta: float,
) -> float:
    return (1 - eps) * exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / sqrt(
        2 * pi * (V + delta_small)
    ) + eps * exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_large))) / sqrt(
        2 * pi * (beta**2 * V + delta_large)
    )


@njit(error_model="numpy", fastmath=True)
def DZ_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_small: float,
    delta_large: float,
    eps: float,
    beta: float,
) -> float:
    small_exponential = exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / sqrt(2 * pi)
    large_exponential = exp(
        -((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_large))
    ) / sqrt(2 * pi)

    return (1 - eps) * small_exponential * (y - omega) / pow(
        V + delta_small, 3 / 2
    ) + eps * beta * large_exponential * (y - beta * omega) / pow(
        beta**2 * V + delta_large, 3 / 2
    )


@njit(error_model="numpy", fastmath=True)
def f_out_Bayes_decorrelated_noise(
    y: float,
    omega: float,
    V: float,
    delta_small: float,
    delta_large: float,
    eps: float,
    beta: float,
) -> float:
    small_exponential = exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
    large_exponential = exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_large)))
    return (
        (y - omega) * (1 - eps) * small_exponential / pow(V + delta_small, 3 / 2)
        + eps
        * beta
        * (y - beta * omega)
        * large_exponential
        / pow(beta**2 * V + delta_large, 3 / 2)
    ) / (
        (1 - eps) * small_exponential / pow(V + delta_small, 1 / 2)
        + eps * large_exponential / pow(beta**2 * V + delta_large, 1 / 2)
    )


# -----------------------------------


@njit(error_model="numpy", fastmath=True)
def f_out_L2(y: float, omega: float, V: float) -> float:
    return (y - omega) / (1 + V)


@njit(error_model="numpy", fastmath=True)
def Df_out_L2(y: float, omega: float, V: float) -> float:
    return -1.0 / (1 + V)


# -----------------------------------


@njit(error_model="numpy", fastmath=True)
def f_out_L1(y: float, omega: float, V: float) -> float:
    return (y - omega + sign(omega - y) * max(abs(omega - y) - V, 0.0)) / V


@vectorize  # @njit(error_model="numpy", fastmath=True)
def Df_out_L1(y: float, omega: float, V: float) -> float:
    if abs(omega - y) > V:
        return 0.0
    else:
        return -1.0 / V


# -----------------------------------


@vectorize
def f_out_Huber(y: float, omega: float, V: float, a: float) -> float:
    if a + a * V + omega < y:
        return a
    elif abs(y - omega) <= a + a * V:
        return (y - omega) / (1 + V)
    elif omega > a + a * V + y:
        return -a
    else:
        return 0.0


@vectorize
def Df_out_Huber(y: float, omega: float, V: float, a: float) -> float:
    if (y < omega and a + a * V + y < omega) or (a + a * V + omega < y):
        return 0.0
    else:
        return -1.0 / (1 + V)
