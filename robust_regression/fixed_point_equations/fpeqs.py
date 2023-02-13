from typing import Tuple
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

        m = damped_update(new_m, m, BLEND_FPE)
        q = damped_update(new_q, q, BLEND_FPE)
        sigma = damped_update(new_sigma, sigma, BLEND_FPE)

        iter_nb += 1
        if iter_nb > max_iter:
            raise ConvergenceError("fixed_point_finder", iter_nb)

    return m, q, sigma


# def _find_fixed_point(alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs):
#     m, q, sigma = fixed_point_finder(
#         var_func,
#         var_hat_func,
#         reg_param=reg_param,
#         alpha=alpha,
#         init=initial_cond,
#         var_hat_kwargs=var_hat_kwargs,
#     )
#     return m, q, sigma


# def fixed_point_func(f, alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs):
#     m, q, sigma = fixed_point_finder(
#         var_func, var_hat_func, reg_param, alpha, initial_cond, var_hat_kwargs
#     )


# def no_parallel_different_alpha_observables_fpeqs_parallel(
#     var_func,
#     var_hat_func,
#     funs=[lambda m, q, sigma: 1 + q - 2 * m],
#     alpha_1=0.01,
#     alpha_2=100,
#     n_alpha_points=16,
#     reg_param=0.1,
#     initial_cond=[0.6, 0.0, 0.0],
#     var_hat_kwargs={},
# ):
#     n_observables = len(funs)
#     alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
#     out_values = np.empty((n_observables, n_alpha_points))
#     results = [None] * len(alphas)

#     for idx, (a, r, k) in enumerate(zip(tqdm(alphas), reg_param, var_hat_kwargs)):
#         results[idx] = _find_fixed_point(a, var_func, var_hat_func, r, initial_cond, k)
#     # inputs = [
#     #     (a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs)
#     #     for a in alphas
#     # ]

#     # with Pool() as pool:
#     #     results = pool.starmap(_find_fixed_point, inputs)

#     for idx, (m, q, sigma) in enumerate(results):
#         fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
#         for jdx, f in enumerate(funs):
#             out_values[jdx, idx] = f(**fixed_point_sols)

#     out_list = [out_values[idx, :] for idx in range(len(funs))]
#     return alphas, out_list


# def no_parallel_different_alpha_observables_fpeqs(
#     var_func,
#     var_hat_func,
#     funs=[lambda m, q, sigma: 1 + q - 2 * m],
#     alpha_1=0.01,
#     alpha_2=100,
#     n_alpha_points=16,
#     reg_param=0.1,
#     initial_cond=[0.6, 0.0, 0.0],
#     var_hat_kwargs={},
# ):
#     n_observables = len(funs)
#     alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
#     out_values = np.empty((n_observables, n_alpha_points))
#     results = [None] * len(alphas)

#     for idx, a in enumerate(tqdm(alphas)):
#         results[idx] = _find_fixed_point(
#             a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
#         )

#     for idx, (m, q, sigma) in enumerate(results):
#         fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
#         for jdx, f in enumerate(funs):
#             out_values[jdx, idx] = f(**fixed_point_sols)

#     out_list = [out_values[idx, :] for idx in range(len(funs))]
#     return alphas, out_list


# def MPI_different_alpha_observables_fpeqs(
#     var_func,
#     var_hat_func,
#     funs=[lambda m, q, sigma: 1 + q - 2 * m],
#     alpha_1=0.01,
#     alpha_2=100,
#     n_alpha_points=16,
#     reg_param=0.1,
#     initial_cond=[0.6, 0.0, 0.0],
#     var_hat_kwargs={},
# ):
#     comm = MPI.COMM_WORLD
#     i = comm.Get_rank()
#     pool_size = comm.Get_size()

#     n_observables = len(funs)
#     alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), pool_size)
#     alpha = alphas[i]
#     out_values = np.empty((n_observables, n_alpha_points))

#     m, q, sigma = _find_fixed_point(
#         alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
#     )

#     ms = np.empty(pool_size)
#     qs = np.empty(pool_size)
#     sigmas = np.empty(pool_size)

#     if i == 0:
#         ms[0] = m
#         qs[0] = q
#         sigmas[0] = sigma

#         for j in range(1, pool_size):
#             ms[j] = comm.recv(source=j)
#         for j in range(1, pool_size):
#             qs[j] = comm.recv(source=j)
#         for j in range(1, pool_size):
#             sigmas[j] = comm.recv(source=j)

#         for idx, (mm, qq, ssigma) in enumerate(zip(ms, qs, sigmas)):
#             fixed_point_sols = {"m": mm, "q": qq, "sigma": ssigma}
#             for jdx, f in enumerate(funs):
#                 out_values[jdx, idx] = f(**fixed_point_sols)

#         out_list = [out_values[idx, :] for idx in range(len(funs))]
#         return alphas, out_list
#     else:
#         print("Process {} sending {}".format(i, reg_param))
#         comm.send(m, dest=0)
#         comm.send(q, dest=0)
#         comm.send(sigma, dest=0)


# def different_alpha_observables_fpeqs(
#     var_func,
#     var_hat_func,
#     funs=[lambda m, q, sigma: 1 + q - 2 * m],
#     alpha_1=0.01,
#     alpha_2=100,
#     n_alpha_points=16,
#     reg_param=0.1,
#     initial_cond=[0.6, 0.0, 0.0],
#     var_hat_kwargs={},
# ):
#     n_observables = len(funs)
#     alphas = np.logspace(np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points)
#     out_values = np.empty((n_observables, n_alpha_points))

#     inputs = [(a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs) for a in alphas]

#     with Pool() as pool:
#         results = pool.starmap(_find_fixed_point, inputs)

#     for idx, (m, q, sigma) in enumerate(results):
#         fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
#         for jdx, f in enumerate(funs):
#             out_values[jdx, idx] = f(**fixed_point_sols)

#     out_list = [out_values[idx, :] for idx in range(len(funs))]
#     return alphas, out_list


# def different_reg_param_gen_error(
#     var_func,
#     var_hat_func,
#     funs=[lambda m, q, sigma: 1 + q - 2 * m],
#     reg_param_1=0.01,
#     reg_param_2=100,
#     n_reg_param_points=16,
#     alpha=0.1,
#     initial_cond=[0.6, 0.0, 0.0],
#     var_hat_kwargs={},
# ):
#     n_observables = len(funs)
#     reg_params = np.logspace(
#         np.log(reg_param_1) / np.log(10),
#         np.log(reg_param_2) / np.log(10),
#         n_reg_param_points,
#     )
#     out_values = np.empty((n_observables, n_reg_param_points))

#     inputs = [
#         (alpha, var_func, var_hat_func, rp, initial_cond, var_hat_kwargs) for rp in reg_params
#     ]

#     with Pool() as pool:
#         results = pool.starmap(_find_fixed_point, inputs)

#     for idx, (m, q, sigma) in enumerate(results):
#         fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
#         for jdx, f in enumerate(funs):
#             out_values[jdx, idx] = f(**fixed_point_sols)

#     out_list = [out_values[idx, :] for idx in range(len(funs))]
#     return reg_params, out_list
