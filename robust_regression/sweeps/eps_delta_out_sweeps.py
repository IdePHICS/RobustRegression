from ..aux_functions.misc import gen_error
from numpy import logspace, empty
from math import log10
from typing import Tuple
from ..fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM
from ..fixed_point_equations.optimality_finding import (
    find_optimal_reg_param_function,
    find_optimal_reg_and_huber_parameter_function,
)


def sweep_eps_delta_out_optimal_lambda_fixed_point(
    var_func,
    var_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_guess_reg_param: float,
    initial_cond_fpe: Tuple[float, float, float],
    funs=[gen_error],
    funs_args=[{}],
    f_min=gen_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
    decreasing=[False, True],
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if len(decreasing) != 2:
        raise ValueError(
            "The length of decreasing should be 2, in this case is {:d}".format(len(decreasing))
        )

    epsilons = (
        logspace(log10(eps_min), log10(eps_max), n_eps_pts)
        if not decreasing[0]
        else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
    )
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing[1]
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )

    n_observables = len(funs)
    f_min_vals = empty((n_eps_pts, n_delta_out_pts))
    reg_params_opt = empty((n_eps_pts, n_delta_out_pts))
    funs_values = [empty((n_eps_pts, n_delta_out_pts)) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()

    old_reg_param_opt_begin_delta_sweep = initial_guess_reg_param
    old_initial_cond_fpe_begin_delta_sweep = initial_cond_fpe
    for idx, eps in enumerate(epsilons):
        old_reg_param_opt = old_reg_param_opt_begin_delta_sweep
        old_initial_cond_fpe = old_initial_cond_fpe_begin_delta_sweep

        for jdx, delta_out in enumerate(delta_outs):
            copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})
            copy_var_hat_func_kwargs.update({"percentage": eps, "delta_out": delta_out})

            (
                f_min_vals[idx, jdx],
                reg_params_opt[idx, jdx],
                (m, q, sigma),
                out_values,
            ) = find_optimal_reg_param_function(
                var_func,
                var_hat_func,
                var_func_kwargs,
                var_hat_func_kwargs,
                initial_guess_reg_param,
                old_initial_cond_fpe,
                funs=funs,
                funs_args=funs_args,
                f_min=f_min,
                f_min_args=f_min_args,
                min_reg_param=min_reg_param,
            )

            old_reg_param_opt = reg_params_opt[idx, jdx]
            old_initial_cond_fpe = (m, q, sigma)

            if jdx == 0:
                old_reg_param_opt_begin_delta_sweep = reg_params_opt[idx, jdx]
                old_initial_cond_fpe_begin_delta_sweep = (m, q, sigma)

            for kdx in range(n_observables):
                funs_values[kdx][idx, jdx] = out_values[kdx]

    # to be checked
    if decreasing[0]:
        f_min_vals = f_min_vals[::-1, :]
        reg_params_opt = reg_params_opt[::-1, :]
        for kdx in range(n_observables):
            funs_values[kdx] = funs_values[kdx][::-1, :]

    if decreasing[1]:
        f_min_vals = f_min_vals[:, ::-1]
        reg_params_opt = reg_params_opt[:, ::-1]
        for kdx in range(n_observables):
            funs_values[kdx] = funs_values[kdx][:, ::-1]

    return epsilons, delta_out, f_min_vals, reg_params_opt, funs_values


def sweep_eps_delta_out_optimal_lambda_hub_param_fixed_point(
    var_func,
    var_hat_func,
    eps_min: float,
    eps_max: float,
    n_eps_pts: int,
    delta_out_min: float,
    delta_out_max: float,
    n_delta_out_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_guess_reg_param: float,
    initial_guess_huber_param: float,
    initial_cond_fpe: Tuple[float, float, float],
    funs=[gen_error],
    funs_args=[{}],
    f_min=gen_error,
    f_min_args={},
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
    decreasing=[False, True],
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if len(decreasing) != 2:
        raise ValueError(
            "The length of decreasing should be 2, in this case is {:d}".format(len(decreasing))
        )

    epsilons = (
        logspace(log10(eps_min), log10(eps_max), n_eps_pts)
        if not decreasing[0]
        else logspace(log10(eps_max), log10(eps_min), n_eps_pts)
    )
    delta_outs = (
        logspace(log10(delta_out_min), log10(delta_out_max), n_delta_out_pts)
        if not decreasing[1]
        else logspace(log10(delta_out_max), log10(delta_out_min), n_delta_out_pts)
    )

    n_observables = len(funs)
    f_min_vals = empty((n_eps_pts, n_delta_out_pts))
    reg_params_opt = empty((n_eps_pts, n_delta_out_pts))
    huber_params_opt = empty((n_eps_pts, n_delta_out_pts))
    funs_values = [empty((n_eps_pts, n_delta_out_pts)) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()
    copy_var_hat_func_kwargs = var_hat_func_kwargs.copy()

    old_reg_param_opt_begin_delta_sweep = initial_guess_reg_param
    old_huber_param_opt_begin_delta_sweep = initial_guess_huber_param
    old_initial_cond_fpe_begin_delta_sweep = initial_cond_fpe
    for idx, eps in enumerate(epsilons):
        old_reg_param_opt = old_reg_param_opt_begin_delta_sweep
        old_huber_param_opt = old_huber_param_opt_begin_delta_sweep
        old_initial_cond_fpe = old_initial_cond_fpe_begin_delta_sweep

        for jdx, delta_out in enumerate(delta_outs):
            copy_var_func_kwargs.update({"reg_param": old_reg_param_opt})
            copy_var_hat_func_kwargs.update(
                {"percentage": eps, "delta_out": delta_out, "a": old_huber_param_opt}
            )

            (
                f_min_vals[idx, jdx],
                (reg_params_opt[idx, jdx], huber_params_opt[idx, jdx]),
                (m, q, sigma),
                out_values,
            ) = find_optimal_reg_and_huber_parameter_function(
                var_func,
                var_hat_func,
                copy_var_func_kwargs,
                copy_var_hat_func_kwargs,
                (old_reg_param_opt, old_huber_param_opt),
                old_initial_cond_fpe,
                funs=funs,
                funs_args=funs_args,
                f_min=f_min,
                f_min_args=f_min_args,
                min_reg_param=min_reg_param,
                min_hub_param=min_huber_param,
            )

            if jdx == 0:
                old_reg_param_opt_begin_delta_sweep = reg_params_opt[idx, jdx]
                old_huber_param_opt_begin_delta_sweep = huber_params_opt[idx, jdx]
                old_initial_cond_fpe_begin_delta_sweep = (m, q, sigma)

            old_reg_param_opt = reg_params_opt[idx, jdx]
            old_huber_param_opt = huber_params_opt[idx, jdx]
            old_initial_cond_fpe = (m, q, sigma)

            for kdx in range(n_observables):
                funs_values[kdx][idx, jdx] = out_values[kdx]

    if decreasing[0]:
        f_min_vals = f_min_vals[::-1, :]
        reg_params_opt = reg_params_opt[::-1, :]
        huber_params_opt = huber_params_opt[::-1, :]
        for kdx in range(n_observables):
            funs_values[kdx] = funs_values[kdx][::-1, :]

    if decreasing[1]:
        f_min_vals = f_min_vals[:, ::-1]
        reg_params_opt = reg_params_opt[:, ::-1]
        huber_params_opt = huber_params_opt[:, ::-1]
        for kdx in range(n_observables):
            funs_values[kdx] = funs_values[kdx][:, ::-1]

    return epsilons, delta_outs, f_min_vals, reg_params_opt, huber_params_opt, funs_values
