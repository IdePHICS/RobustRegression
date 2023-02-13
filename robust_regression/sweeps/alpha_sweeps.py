from numpy import logspace, empty
from math import log10
from typing import Tuple
from ..fixed_point_equations.fpeqs import fixed_point_finder
from ..aux_functions.misc import gen_error
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)


def sweep_alpha_fixed_point(
    var_func,
    var_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[gen_error],
    funs_args=[list()],
    decreasing=False,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    out_list = [empty(n_alpha_pts) for _ in range(n_observables)]
    ms_qs_sigmas = empty((n_alpha_pts, 3))

    old_initial_cond = initial_cond_fpe
    for idx, alpha in enumerate(alphas):
        var_hat_func_kwargs.update({"alpha": alpha})
        ms_qs_sigmas[idx] = fixed_point_finder(
            var_func, var_hat_func, old_initial_cond, var_func_kwargs, var_hat_func_kwargs
        )
        old_initial_cond = tuple(ms_qs_sigmas[idx])
        m, q, sigma = ms_qs_sigmas[idx]
        for jdx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_list[jdx][idx] = f(m, q, sigma, *f_args)

    if decreasing:
        alphas = alphas[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return alphas, out_list


def sweep_alpha_optimal_lambda_fixed_point(
    var_func,
    var_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    inital_guess_lambda: float,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[gen_error],
    funs_args=[list()],
    f_min=gen_error,
    f_min_args=(),
    min_reg_param=SMALLEST_REG_PARAM,
    decreasing=False,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    f_min_vals = empty(n_alpha_pts)
    reg_params_opt = empty(n_alpha_pts)
    funs_values = [empty(n_alpha_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_lambda
    for idx, alpha in enumerate(alphas):
        copy_var_hat_func_kwargs.update({"alpha": float(alpha)})
        copy_var_func_kwargs.update({"reg_param": float(old_reg_param_opt)})

        (
            f_min_vals[idx],
            reg_params_opt[idx],
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_param_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            old_reg_param_opt,
            old_initial_cond_fpe,
            funs=funs,
            funs_args=funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
        )
        old_reg_param_opt = reg_params_opt[idx]
        old_initial_cond_fpe = (m, q, sigma)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        alphas = alphas[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return alphas, f_min_vals, reg_params_opt, funs_values


def sweep_alpha_optimal_lambda_hub_param_fixed_point(
    var_func,
    var_hat_func,
    alpha_min: float,
    alpha_max: float,
    n_alpha_pts: int,
    inital_guess_params: Tuple[float, float],
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond_fpe=(0.6, 0.01, 0.9),
    funs=[gen_error],
    funs_args=[list()],
    f_min=gen_error,
    f_min_args=(),
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=False,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if alpha_min > alpha_max:
        raise ValueError(
            "alpha_min should be smaller than alpha_max, in this case are {:f} and {:f}".format(
                alpha_min, alpha_max
            )
        )

    n_observables = len(funs)
    alphas = (
        logspace(log10(alpha_min), log10(alpha_max), n_alpha_pts)
        if not decreasing
        else logspace(log10(alpha_max), log10(alpha_min), n_alpha_pts)
    )
    f_min_vals = empty(n_alpha_pts)
    reg_params_opt = empty(n_alpha_pts)
    hub_params_opt = empty(n_alpha_pts)
    funs_values = [empty(n_alpha_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()

    old_initial_cond_fpe = initial_cond_fpe
    old_reg_param_opt = inital_guess_params[0]
    old_hub_param_opt = inital_guess_params[1]
    for idx, alpha in enumerate(alphas):
        copy_var_hat_func_kwargs.update({"alpha": alpha, "a": old_hub_param_opt})
        copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})

        (
            f_min_vals[idx],
            (reg_params_opt[idx], hub_params_opt[idx]),
            (m, q, sigma),
            out_values,
        ) = find_optimal_reg_and_huber_parameter_function(
            var_func,
            var_hat_func,
            copy_var_func_kwargs,
            copy_var_hat_func_kwargs,
            (old_reg_param_opt, old_hub_param_opt),
            old_initial_cond_fpe,
            funs=funs,
            funs_args=funs_args,
            f_min=f_min,
            f_min_args=f_min_args,
            min_reg_param=min_reg_param,
            min_huber_param=min_huber_param,
        )

        old_reg_param_opt = reg_params_opt[idx]
        old_hub_param_opt = hub_params_opt[idx]
        old_initial_cond_fpe = (m, q, sigma)

        for jdx in range(n_observables):
            funs_values[jdx][idx] = out_values[jdx]

    if decreasing:
        alphas = alphas[::-1]
        f_min_vals = f_min_vals[::-1]
        reg_params_opt = reg_params_opt[::-1]
        for idx, fun_vals in enumerate(funs_values):
            funs_values[idx] = fun_vals[::-1]

    return alphas, f_min_vals, (reg_params_opt, hub_params_opt), funs_values
