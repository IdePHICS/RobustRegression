import robust_regression.sweeps.delta_out_sweep as dosw
import matplotlib.pyplot as plt
from robust_regression.fixed_point_equations.fpe_L2_loss import (
    var_hat_func_L2_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_L1_loss import (
    var_hat_func_L1_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_Huber_loss import (
    var_hat_func_Huber_decorrelated_noise,
)
from robust_regression.fixed_point_equations.fpe_BO import var_func_BO, var_hat_func_BO_num_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_L2_regularization import var_func_L2
import numpy as np
from robust_regression.aux_functions.misc import gen_error_ML, gen_error, gen_error_ML_BO
from robust_regression.aux_functions.stability_functions import (
    stability_ridge,
    stability_l1_l2,
    stability_huber,
)


def sigma_order_param(m, q, sigma):
    return sigma


def q_order_param(m, q, sigma):
    return q


def m_order_param(m, q, sigma):
    return m


alpha, delta_in, percentage, beta = 10.0, 1.0, 0.1, 0.0
delta_out_min, delta_out_max, n_delta_out_pts = 0.01, 10, 100
# delta_out_init = 100

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out_max * q:
        initial_condition = [m, q, sigma]
        break

print("begin")

delta_outs, e_gen_l2, reg_params_opt_l2, (ms_l2, qs_l2, sigmas_l2) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    0.1,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{},{},{}],
    decreasing=True,
)

print("L2 done")

_, e_gen_l1, reg_params_opt_l1, (ms_l1, qs_l1, sigmas_l1) = dosw.sweep_delta_out_optimal_lambda_fixed_point(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    0.5,
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{},{},{}],
    decreasing=True,
)

print("L1 done")

(
    _,
    e_gen_hub,
    (reg_params_opt_hub, hub_params_opt),
    (ms_hub, qs_hub, sigmas_hub),
) = dosw.sweep_delta_out_optimal_lambda_hub_param_fixed_point(
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    n_delta_out_pts,
    [0.5, 1.0],
    {"reg_param": 3.0},
    {
        "alpha": alpha,
        "delta_in": delta_in,
        "delta_out": delta_out_max,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    },
    initial_cond_fpe=initial_condition,
    f_min=gen_error_ML,
    f_min_args={"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    update_f_min_args=True,
    funs=[m_order_param, q_order_param, sigma_order_param],
    funs_args=[{},{},{}],
    decreasing=True,
)

print("Huber done")

delta_outs_BO, (gen_error_BO, qs_BO) = dosw.sweep_delta_out_fixed_point(
    var_func_BO,
    var_hat_func_BO_num_decorrelated_noise,
    delta_out_min,
    delta_out_max,
    30,
    {"reg_param": 3.0},
    {"alpha": alpha, "delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta},
    initial_cond=(0.6, 0.01, 0.9),
    funs=[gen_error_ML_BO, q_order_param],
    funs_args=[{"delta_in": delta_in, "delta_out": 10.0, "percentage": percentage, "beta": beta}, {}],
    update_funs_args=[True, False],
    decreasing=True,
)

print("BO done")

# ----------------------------

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.title(
    r"$\alpha = {}$, $\beta = {}$, $\epsilon = {}$, $\Delta_{{in}} = {}$".format(alpha, beta, percentage, delta_in)
)
plt.plot(delta_outs, e_gen_l2, label="L2")
plt.plot(delta_outs, e_gen_l1, label="L1")
plt.plot(delta_outs, e_gen_hub, label="Huber")
plt.plot(delta_outs_BO, gen_error_BO, label="BO")
plt.xlabel(r"$\Delta_{out}$")
plt.ylabel(r"$E_{gen}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(delta_outs, reg_params_opt_l2, label="L2")
plt.plot(delta_outs, reg_params_opt_l1, label="L1")
plt.plot(delta_outs, reg_params_opt_hub, label="Huber lambda")
plt.plot(delta_outs, hub_params_opt, label="Huber a")
plt.ylim([0, 8])
plt.xlabel(r"$\Delta_{out}$")
plt.ylabel(r"$\lambda_{opt}$")
plt.xscale("log")
plt.legend()
plt.grid()


plt.show()
