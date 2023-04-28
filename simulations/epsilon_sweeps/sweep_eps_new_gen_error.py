import robust_regression.sweeps.alpha_sweeps as alsw
import robust_regression.sweeps.eps_sweep as epsw
import matplotlib.pyplot as plt
from robust_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_BO import var_func_BO, var_hat_func_BO_num_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_L1_loss import (
    var_hat_func_L1_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
from robust_regression.aux_functions.misc import gen_error, gen_error_ML, gen_error_ML_BO
import numpy as np


def condition_MP(alphas):
    return -((1 - np.sqrt(alphas)) ** 2)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, delta_out, beta = 10.0, 0.1, 5.0, 0.0
eps_min, eps_max, n_eps_pts = 0.0001, 0.5, 100

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        initial_condition = [m, q, sigma]
        break

epsilons, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = epsw.sweep_eps_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    delta_in,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
)

print("L2 done")

_, e_gen_l1, reg_params_opt_l1, (ms_l1, qs_l1, sigmas_l1) = epsw.sweep_eps_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    0.5,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
)

print("L1 done")

(
    _,
    e_gen_hub,
    (reg_params_opt_hub, hub_params_opt),
    (ms_hub, qs_hub, sigmas_hub),
) = epsw.sweep_eps_optimal_lambda_hub_param_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    eps_min,
    eps_max,
    n_eps_pts,
    [0.5, 1.0],
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{}, {}, {}],
)

print("Huber done")

(
    epsilons_BO,
    (gen_error_BO, qs_BO),
) = epsw.sweep_eps_fixed_point(
    var_func_BO,
    var_hat_func_BO_num_decorrelated_noise,
    eps_min,
    eps_max,
    30,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out,
        "percentage": 0.3,
        "beta": beta,
    },
    initial_cond=(0.6, 0.01, 0.9),
    funs=[gen_error_ML_BO, q_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": delta_out, "percentage": 0.3, "beta": beta}, {}],
    update_funs_args=[True, False],
    decreasing=False,
)

print("BO done")

# ----------------------------

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    r"$\alpha = {}$, $\beta = {}$, $\Delta_{{in}} = {}$, $\Delta_{{in}} = {}$".format(alpha, beta, delta_in, delta_out)
)
plt.plot(epsilons, e_gen_l2, label="L2")
plt.plot(epsilons, e_gen_l1, label="L1")
plt.plot(epsilons, e_gen_hub, label="Huber")
plt.plot(epsilons_BO, gen_error_BO, label="BO")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$E_{gen}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(epsilons, reg_params_opt_l2, label="L2")
plt.plot(epsilons, reg_params_opt_l1, label="L1")
plt.plot(epsilons, reg_params_opt_hub, label="Huber lamb")
plt.plot(epsilons, hub_params_opt, label="Huber param")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\lambda_{opt}$")
plt.xscale("log")
plt.legend()
plt.grid()


plt.show()
