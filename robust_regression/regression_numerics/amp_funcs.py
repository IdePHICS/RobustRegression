# import numpy as np
from numpy import pi, ndarray, mean, abs, amax, zeros
from math import exp, sqrt
from numpy.random import random
from numba import njit

# from ..utils.integration_utils import x_ge, w_ge
from ..regression_numerics import TOL_GAMP, BLEND_GAMP, MAX_ITER_GAMP
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update


def GAMP_algorithm_unsimplified(
    f_w: callable,
    Df_w: callable,
    f_out: callable,
    Df_out: callable,
    ys: ndarray,
    xs: ndarray,
    f_w_args: tuple,
    f_out_args: tuple,
    init_w_hat,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP,
):
    n, d = xs.shape

    # random init
    w_hat_t = init_w_hat  # 0.1 * random(d) + 0.95
    c_w_t = zeros(d) # 0.1 * random(d) + 0.01
    f_out_t_1 = zeros(n) # 0.5 * random(n) + 0.001

    F = xs / sqrt(d)
    F2 = F**2

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = (F @ w_hat_t) - (V_t * f_out_t_1)

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = (f_out_t @ F) + (Lambda_t * w_hat_t)

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = mean(abs(new_w_hat_t - w_hat_t))

        if iter_nb % 500 == 0:
            print(err)      
        
        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("GAMP_algorithm", iter_nb)

    return w_hat_t