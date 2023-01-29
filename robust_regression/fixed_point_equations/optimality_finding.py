import numpy as np
from numba import njit
from scipy.optimize import minimize
from fpeqs import fixed_point_finder
from fixed_point_equations import SMALLEST_REG_PARAM, SMALLEST_HUBER_PARAM, XATOL, FATOL

# --------------------------------


@njit(error_model="numpy", fastmath=True)
def gen_error(m, q, sigma, *args):
    return 1 + q - 2 * m


# --------------------------------


def find_optimal_reg_param_function(
    alpha,
    var_func,
    var_hat_func,
    var_hat_kwargs,
    initial_guess,
    initial_cond_fpe,
    f=gen_error,
    args=(),
    min_reg_param=SMALLEST_REG_PARAM,
):
    def minimize_fun(reg_param):
        m, q, sigma = fixed_point_finder(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial_cond_fpe,
            var_hat_kwargs=var_hat_kwargs,
        )
        return f(m, q, sigma, *args)

    bnds = [(min_reg_param, None)]
    obj = minimize(
        minimize_fun,
        x0=initial_guess,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )
    if obj.success:
        fun_val = obj.fun
        reg_param_opt = obj.x
        return fun_val, reg_param_opt
    else:
        raise RuntimeError("Minima could not be found.")


def find_optimal_reg_and_huber_parameter_function(
    alpha,
    var_func,
    var_hat_func,
    var_hat_kwargs,
    inital_guess,
    initial_cond_fpe,
    f=gen_error,
    args=(),
    min_reg_param=SMALLEST_REG_PARAM,
    min_huber_param=SMALLEST_HUBER_PARAM,
):
    def minimize_fun(x):
        reg_param, a = x
        var_hat_kwargs.update({"a": a})
        m, q, sigma = fixed_point_finder(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial_cond_fpe,
            var_hat_kwargs=var_hat_kwargs,
        )
        return f(m, q, sigma, *args)

    bnds = [(min_reg_param, None), (min_huber_param, None)]
    obj = minimize(
        minimize_fun,
        x0=inital_guess,
        method="Nelder-Mead",
        bounds=bnds,
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )
    if obj.success:
        fun_val = obj.fun
        reg_param_opt, a_opt = obj.x
        return fun_val, reg_param_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")

