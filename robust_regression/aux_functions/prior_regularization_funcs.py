from numba import njit
from math import abs, exp


@njit(error_model="numpy", fastmath=True)
def Z_w_L2_regularization(gamma, Lambda, reg_param):
    return exp((gamma**2 * Lambda) / (2 * (reg_param + Lambda) ** 2))


@njit(error_model="numpy", fastmath=True)
def f_w_L2_regularization(gamma, Lambda, reg_param):
    return gamma / (reg_param + Lambda)


@njit(error_model="numpy", fastmath=True)
def Df_w_L2_regularization(gamma, Lambda, reg_param):
    return 1 / (reg_param + Lambda)


# -------------------------


@njit(error_model="numpy", fastmath=True)
def f_w_L1_regularization(gamma, Lambda, reg_param):
    if gamma > reg_param:
        return (gamma - reg_param) / Lambda
    elif gamma + reg_param < 0:
        return (gamma + reg_param) / Lambda
    else:
        return 0.0


@njit(error_model="numpy", fastmath=True)
def Df_w_L1_regularization(gamma, Lambda, reg_param):
    if abs(gamma) > reg_param:
        return 1 / Lambda
    else:
        return 0.0
