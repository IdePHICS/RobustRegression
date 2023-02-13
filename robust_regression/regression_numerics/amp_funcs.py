import numpy as np
from numpy import pi
from math import exp, sqrt
from numpy.random import random
from numba import njit
from ..utils.integration_utils import x_ge, w_ge
from ..regression_numerics import TOL_GAMP, BLEND_GAMP, MAX_ITER_GAMP
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update


def GAMP_algorithm_unsimplified(
    f_w : callable,
    Df_w : callable,
    f_out : callable,
    Df_out : callable,
    ys : np.ndarray,
    xs : np.ndarray,
    f_w_args : tuple,
    f_out_args : tuple,
    init_w_hat,
    abs_tol=TOL_GAMP,
    max_iter=MAX_ITER_GAMP,
    blend=BLEND_GAMP
):
    n, d = xs.shape

    # random init
    w_hat_t = init_w_hat  # 0.1 * random(d) + 0.95
    c_w_t = 0.1 * random(d) + 0.01
    # probably different dimension
    f_out_t_1 = 0.5 * random(n) + 0.001

    # print(w_hat_t.shape, c_w_t.shape, f_out_t_1.shape, xs.shape)

    F = xs / sqrt(d)
    F2 = F**2

    err = 1.0
    iter_nb = 0
    while err > abs_tol:
        V_t = F2 @ c_w_t
        omega_t = F @ w_hat_t - V_t * f_out_t_1

        f_out_t = f_out(ys, omega_t, V_t, *f_out_args)
        Df_out_t = Df_out(ys, omega_t, V_t, *f_out_args)

        Lambda_t = -Df_out_t @ F2
        gamma_t = f_out_t @ F + Lambda_t * w_hat_t

        new_w_hat_t = f_w(gamma_t, Lambda_t, *f_w_args)
        new_c_w_t = Df_w(gamma_t, Lambda_t, *f_w_args)

        err = max(np.max(np.abs(new_w_hat_t - w_hat_t)), np.max(np.abs(new_c_w_t - c_w_t)))

        w_hat_t = damped_update(new_w_hat_t, w_hat_t, blend)
        c_w_t = damped_update(new_c_w_t, c_w_t, blend)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("GAMP_algorithm", iter_nb)

    return w_hat_t


# to be changed
# @njit(error_model="numpy")
def find_coefficients_AMP(input_funs, output_funs, ys, xs, *noise_args):
    _, d = xs.shape

    a_t_1 = 0.1 * np.random.rand(d) + 0.95
    v_t_1 = 0.5 * np.random.rand(d) + 0.01
    gout_t_1 = 0.5 * np.random.rand(1) + 0.001

    F = xs / sqrt(d)
    F2 = F**2

    err = 1.0
    while err > TOL_GAMP:
        V_t = F2 @ v_t_1
        omega_t = F @ a_t_1 - V_t * gout_t_1

        gout_t, Dgout_t = output_funs(ys, omega_t, V_t, *noise_args)

        sigma_t = -1 / (Dgout_t @ F2)
        R_t = a_t_1 + sigma_t * (gout_t @ F)

        a_t, v_t = input_funs(R_t, sigma_t)

        err = max(np.max(a_t - a_t_1), np.max(v_t - v_t_1))

        a_t_1 = BLEND_GAMP * a_t + (1 - BLEND_GAMP) * a_t_1
        v_t_1 = BLEND_GAMP * v_t + (1 - BLEND_GAMP) * v_t_1
        gout_t_1 = BLEND_GAMP * gout_t + (1 - BLEND_GAMP) * gout_t_1

    return a_t  # , v_t


@njit(error_model="numpy")
def gaussian_prior(x):
    return exp(-0.5 * x**2) / sqrt(2 * pi)


@njit(error_model="numpy")
def input_functions_gaussian_prior(Rs, sigmas):
    fa = np.empty_like(sigmas)
    fv = np.empty_like(sigmas)

    for idx, (sigma, R) in enumerate(zip(sigmas, Rs)):
        z = sqrt(2.0) * sqrt(sigma) * x_ge + R
        jacobian = sqrt(2.0) * sqrt(sigma)

        simple_int = np.sum(w_ge * jacobian * gaussian_prior(z))
        fa[idx] = np.sum(w_ge * jacobian * gaussian_prior(z) * z) / (simple_int)

        first_term_fv = np.sum(w_ge * jacobian * z * (z - R) * gaussian_prior(z)) / simple_int
        second_term_fv = (
            fa[idx] * np.sum(w_ge * jacobian * (z - R) * gaussian_prior(z)) / simple_int
        )
        fv[idx] = first_term_fv - second_term_fv

    return fa, fv


@njit(error_model="numpy")
def likelihood_single_gaussians(y, z, delta):
    return exp(-0.5 * (y - z) ** 2 / delta) / (sqrt(2 * pi * delta))


@njit(error_model="numpy")
def output_functions_single_noise(ys, omegas, Vs, delta):
    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = sqrt(2.0) * sqrt(V) * x_ge + omega
        jacobian = sqrt(2.0) * sqrt(V)

        simple_int = np.sum(w_ge * jacobian * likelihood_single_gaussians(y, z, delta))
        gout[idx] = np.sum(
            w_ge * jacobian * likelihood_single_gaussians(y, z, delta) * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge * jacobian * likelihood_single_gaussians(y, z, delta) * (z - omega) ** 2
        ) / (V**2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout


@njit(error_model="numpy")
def likelihood_double_gaussians(y, z, delta_in, delta_out, eps):
    return (1 - eps) / (sqrt(2 * pi * delta_in)) * exp(-0.5 * (y - z) ** 2 / (delta_in)) + eps / (
        sqrt(2 * pi * delta_out)
    ) * exp(-0.5 * (y - z) ** 2 / (delta_out))


@njit(error_model="numpy")
def output_functions_double_noise(ys, omegas, Vs, delta_in, delta_out, eps):
    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = sqrt(2.0) * sqrt(V) * x_ge + omega
        jacobian = sqrt(2.0) * sqrt(V)

        simple_int = np.sum(
            w_ge * jacobian * likelihood_double_gaussians(y, z, delta_in, delta_out, eps)
        )
        gout[idx] = np.sum(
            w_ge
            * jacobian
            * likelihood_double_gaussians(y, z, delta_in, delta_out, eps)
            * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge
            * jacobian
            * likelihood_double_gaussians(y, z, delta_in, delta_out, eps)
            * (z - omega) ** 2
        ) / (V**2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout


@njit(error_model="numpy")
def likelihood_decorrelated_gaussians(
    y: float, z: float, delta_in: float, delta_out: float, eps: float, beta: float
) -> float:
    return (1 - eps) / (sqrt(2 * pi * delta_in)) * exp(-0.5 * (y - z) ** 2 / (delta_in)) + eps / (
        sqrt(2 * pi * delta_out)
    ) * exp(-0.5 * (y - beta * z) ** 2 / (delta_out))


@njit(error_model="numpy")
def output_functions_decorrelated_noise(ys, omegas, Vs, delta_in, delta_out, eps, beta):
    gout = np.empty_like(ys)
    Dgout = np.empty_like(ys)

    for idx, (y, omega, V) in enumerate(zip(ys, omegas, Vs)):
        z = sqrt(2.0) * sqrt(V) * x_ge + omega
        jacobian = sqrt(2.0) * sqrt(V)

        simple_int = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_in, delta_out, eps, beta)
        )
        gout[idx] = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_in, delta_out, eps, beta)
            * (z - omega)
        ) / (V * simple_int)

        first_term_Dgout = np.sum(
            w_ge
            * jacobian
            * likelihood_decorrelated_gaussians(y, z, delta_in, delta_out, eps, beta)
            * (z - omega) ** 2
        ) / (V**2 * simple_int)
        Dgout[idx] = first_term_Dgout - 1 / V - gout[idx] ** 2

    return gout, Dgout
