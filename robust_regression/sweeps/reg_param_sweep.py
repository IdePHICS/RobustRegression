from numpy import logspace, linspace, empty
from math import log10
from ..aux_functions.misc import gen_error
from ..fixed_point_equations.fpeqs import fixed_point_finder


def sweep_reg_param_fixed_point(
    var_func,
    var_hat_func,
    reg_param_min: float,
    reg_param_max: float,
    n_reg_param_pts: int,
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    initial_cond=(0.6, 0.01, 0.9),
    funs=[gen_error],
    funs_args=[list()],
    linear=False,
    decreasing=True,
):
    if len(funs) != len(funs_args):
        raise ValueError(
            "The length of funs and funs_args should be the same, in this case is {:d} and {:d}".format(
                len(funs), len(funs_args)
            )
        )

    if reg_param_min > reg_param_max:
        raise ValueError(
            "reg_param_min should be smaller than reg_param_max, in this case are {:f} and {:f}".format(
                reg_param_min, reg_param_max
            )
        )

    if not linear:
        if reg_param_min <= 0.0 or reg_param_max <= 0.0:
            raise ValueError(
                "reg_param_min and reg_param_max should be positive in this case are {:f} and {:f}".format(
                    reg_param_min, reg_param_max
                )
            )

    n_observables = len(funs)
    if linear:
        reg_params = (
            linspace(reg_param_min, reg_param_max, n_reg_param_pts)
            if not decreasing
            else linspace(reg_param_min, reg_param_max, n_reg_param_pts)
        )
    else:
        reg_params = (
            logspace(log10(reg_param_min), log10(reg_param_max), n_reg_param_pts)
            if not decreasing
            else logspace(log10(reg_param_min), log10(reg_param_max), n_reg_param_pts)
        )
    out_list = [empty(n_reg_param_pts) for _ in range(n_observables)]

    copy_var_func_kwargs = var_func_kwargs.copy()

    old_initial_cond = initial_cond
    for idx, reg_param in enumerate(reg_params):
        copy_var_func_kwargs.update({"reg_param": reg_param})

        m, q, sigma = fixed_point_finder(
            var_func, var_hat_func, old_initial_cond, copy_var_func_kwargs, var_hat_func_kwargs
        )

        old_initial_cond = tuple([m, q, sigma])

        for jdx, (f, f_args) in enumerate(zip(funs, funs_args)):
            out_list[jdx][idx] = f(m, q, sigma, *f_args)

    if decreasing:
        reg_params = reg_params[::-1]
        for idx, obs_vals in enumerate(out_list):
            out_list[idx] = obs_vals[::-1]

    return reg_params, out_list
