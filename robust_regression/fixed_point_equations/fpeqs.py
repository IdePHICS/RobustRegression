from typing import Tuple
from numba import njit
from numba.typed import Dict
from ..fixed_point_equations import BLEND_FPE, TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE
from ..utils.errors import ConvergenceError
from ..aux_functions.misc import damped_update


# def fixed_point_finder(
#     var_func,
#     var_hat_func,
#     reg_param: float,
#     alpha: float,
#     initial_condition: Tuple[float, float, float],
#     var_hat_args: tuple,
# ):
#     m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
#     err = 1.0
#     iter_nb = 0
#     while err > TOL_FPE:
#         m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, *var_hat_args)

#         new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, reg_param)

#         err = np.max(np.abs([(new_m - m), (new_q - q), (new_sigma - sigma)]))

#         m = damped_update(new_m, m, BLEND_FPE)
#         q = damped_update(new_q, q, BLEND_FPE)
#         sigma = damped_update(new_sigma, sigma, BLEND_FPE)

#         iter_nb += 1
#         if iter_nb > MAX_ITER_FPE:
#             raise ConvergenceError("fixed_point_finder", iter_nb)

#     return m, q, sigma


def fixed_point_finder(
    var_func,
    var_hat_func,
    initial_condition: Tuple[float, float, float],
    var_func_kwargs: dict,
    var_hat_func_kwargs: dict,
    abs_tol: float = TOL_FPE,
    min_iter: int = MIN_ITER_FPE,
    max_iter: int = MAX_ITER_FPE,
):
    m, q, sigma = initial_condition[0], initial_condition[1], initial_condition[2]
    err = 1.0
    iter_nb = 0
    while err > abs_tol or iter_nb < min_iter:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, **var_hat_func_kwargs)

        new_m, new_q, new_sigma = var_func(m_hat, q_hat, sigma_hat, **var_func_kwargs)

        err = max([abs(new_m - m), abs(new_q - q), abs(new_sigma - sigma)])

        # if iter_nb % 100 == 0:
        #     print("\t", err)

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    return m, q, sigma