import numpy as np
from numba import njit
from amp_funcs import (
    input_functions_gaussian_prior,
    output_functions_decorrelated_noise,
    output_functions_double_noise,
    output_functions_single_noise,
)

from numerics import TOL_GAMP, BLEND_GAMP

def find_numerical_mean_std(
    alpha,
    measure_fun,
    find_coefficients_fun,
    n_features,
    repetitions,
    measure_fun_args,
    find_coefficients_fun_args,
):
    all_gen_errors = np.empty((repetitions,))

    for idx in range(repetitions):
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(np.around(n_features * alpha)), 1),
            n_generalization=1,
            measure_fun_args=measure_fun_args,
        )
        
        print(xs.shape, ys.shape)

        estimated_theta = find_coefficients_fun(ys, xs, *find_coefficients_fun_args)

        all_gen_errors[idx] = np.divide(
            np.sum(np.square(ground_truth_theta - estimated_theta)), n_features
        )

        del xs
        del ys
        del ground_truth_theta

    error_mean, error_std = np.mean(all_gen_errors), np.std(all_gen_errors)
    print(alpha, "Done.")

    del all_gen_errors

    return error_mean, error_std


@njit(error_model="numpy", fastmath=True)
def find_coefficients_AMP(input_funs, output_funs, ys, xs, *noise_args):
    _, d = xs.shape

    a_t_1 = 0.1 * np.random.rand(d) + 0.95
    v_t_1 = 0.5 * np.random.rand(d) + 0.01
    gout_t_1 = 0.5 * np.random.rand(1) + 0.001

    F = xs / np.sqrt(d)
    F2 = F ** 2

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


def find_coefficients_AMP_single_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior, output_functions_single_noise, ys, xs, *noise_args
    )


def find_coefficients_AMP_double_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior, output_functions_double_noise, ys, xs, *noise_args
    )


def find_coefficients_AMP_decorrelated_noise(ys, xs, *noise_args):
    return find_coefficients_AMP(
        input_functions_gaussian_prior,
        output_functions_decorrelated_noise,
        ys,
        xs,
        *noise_args
    )

