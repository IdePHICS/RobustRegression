import robust_regression.regression_numerics.amp_funcs as amp
from robust_regression.sweeps.q_sweeps import sweep_fw_first_arg_GAMP
import robust_regression.regression_numerics.data_generation as dg
from robust_regression.aux_functions.prior_regularization_funcs import (
    f_w_projection_on_sphere,
    Df_w_projection_on_sphere,
)
from robust_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from robust_regression.fixed_point_equations.fpe_projection_denoising import (
    var_func_projection_denoising,
)
from robust_regression.aux_functions.misc import damped_update
from robust_regression.aux_functions.likelihood_channel_functions import f_out_L1, Df_out_L1
from robust_regression.aux_functions.loss_functions import l1_loss
from robust_regression.aux_functions.stability_functions import stability_l1_l2, stability_huber, stability_ridge
from robust_regression.aux_functions.training_errors import training_error_l1_loss
import numpy as np
import matplotlib.pyplot as plt
import robust_regression.fixed_point_equations as fpe
from robust_regression.utils.errors import ConvergenceError

print("here")

abs_tol = 1e-8
min_iter = 100
max_iter = 10000
blend = 1.0

f_out_args = (1.0, 5.0, 0.3, 0.0)

n_features = 1000
repetitions = 5

alpha, delta_in, delta_out, percentage, beta = 2.0, 1.0, 5.0, 0.3, 0.0
n_samples = max(int(np.around(n_features * alpha)), 1)

gen_err_q = list()
gen_err_mean = list()
gen_err_std = list()
train_err_mean = list()
train_err_std = list()

qs_amp = np.logspace(-1, 1.5, 15)

gen_err_q, (gen_err_mean,), (gen_err_std,) = sweep_fw_first_arg_GAMP(
    f_w_projection_on_sphere,
    Df_w_projection_on_sphere,
    f_out_L1,
    Df_out_L1,
    dg.measure_gen_decorrelated,
    alpha,
    0.1,
    10**1.5,
    15,
    repetitions,
    n_features,
    tuple(),
    (delta_in, delta_out, percentage, beta),
    funs=[gen_error],
    funs_args=[list()],
    blend=0.85
)

for idx, q in enumerate(qs_amp):
    print("idx = ", idx, " q = ", q)

    all_gen_errors = np.empty(repetitions)
    all_train_errors = np.empty(repetitions)
    try:
        for jdx in range(repetitions):
            print("Iteration ", jdx, "/", repetitions)

            xs, ys, _, _, theta_0_teacher = dg.data_generation(
                dg.measure_gen_decorrelated,
                n_features,
                n_samples,
                1,
                (delta_in, delta_out, percentage, beta),
            )

            f_w_args = (float(q),)
            f_out_args = tuple()

            estimated_theta = amp.GAMP_algorithm_unsimplified(
                f_w_projection_on_sphere,
                Df_w_projection_on_sphere,
                f_out_L1,
                Df_out_L1,
                ys,
                xs,
                f_w_args,
                f_out_args,
                theta_0_teacher,
                blend=0.65,
            )

            print("Norm of the estimated_theta: ", np.linalg.norm(estimated_theta), " value of sqrt(q): ", np.sqrt(q))

            # this stores all the generalization errors
            all_gen_errors[jdx] = np.divide(
                np.sum(np.square(theta_0_teacher - estimated_theta)), n_features
            )

            # this stores all the training errors
            all_train_errors[jdx] = np.divide(
                np.sum(l1_loss(ys, np.dot(np.divide(xs, np.sqrt(n_features)), estimated_theta))),
                n_samples,
            )

            del xs
            del ys
            del theta_0_teacher

        gen_err_q.append(float(q))
        gen_err_mean.append(np.mean(all_gen_errors))
        gen_err_std.append(np.std(all_gen_errors))

        train_err_mean.append(np.mean(all_train_errors))
        train_err_std.append(np.std(all_train_errors))

    except ConvergenceError as e:
        print(e)
        break


qs = np.logspace(-1, 1.5, 500)
ms = np.empty_like(qs)
sigmas = np.empty_like(qs)
m_hats = np.empty_like(qs)
q_hats = np.empty_like(qs)
sigma_hats = np.empty_like(qs)
training_error = np.empty_like(qs)


q = qs[0]
while True:
    m = 10 * np.random.random() + 0.01
    sigma = 10 * np.random.random() + 0.01
    if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
        break
for idx, q in enumerate(qs):
    try:
        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, sigma_hat = var_hat_func_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, new_q, new_sigma = var_func_projection_denoising(m_hat, q_hat, sigma_hat, q)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend)
            sigma = damped_update(new_sigma, sigma, blend)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        ms[idx] = m
        sigmas[idx] = sigma
        m_hats[idx] = m_hat
        sigma_hats[idx] = sigma_hat
        q_hats[idx] = q_hat

        training_error[idx] = training_error_l1_loss(
            m, q, sigma, delta_in, delta_out, percentage, beta
        )
    except (ConvergenceError, ValueError) as e:
        ms[idx:] = np.nan
        sigmas[idx:] = np.nan
        m_hats[idx:] = np.nan
        sigma_hats[idx:] = np.nan
        q_hats[idx:] = np.nan
        training_error[idx:] = np.nan
        break

# save the results of the AMP experiment in a csv file in the folder simulations/data
np.savetxt(
    "./simulations/data/projection_amp_sweep_q_L1.csv",
    np.array([gen_err_q, gen_err_mean, gen_err_std, train_err_mean, train_err_std]).T,
    delimiter=",",
    header="q,gen_err_mean,gen_err_std,train_err_mean,train_err_std",
)

plt.figure(figsize=(10, 7.5))
plt.title(
    "L1 loss Projective Denoiser"
    + " d = {:d}".format(n_features)
    + r"$\alpha$ = "
    + "{:.2f}".format(alpha)
    + r"$\Delta_{in}$ = "
    + "{:.2f}".format(delta_in)
    + r" $\Delta_{out}$ ="
    + "{:.2f}".format(delta_out)
    + r"$\epsilon$ = "
    + "{:.2f}".format(percentage)
    + r" $\alpha$ = "
    + "{:.2f}".format(alpha)
    + r" $\beta$ = "
    + "{:.2f}".format(beta)
)

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    gen_err_q,
    gen_err_mean,
    yerr=gen_err_std,
    label="AMP Generalization Error",
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, 1 + qs - 2 * ms, label="Theoretical Generalization Error", color=color, linestyle="-")

color = next(plt.gca()._get_lines.prop_cycler)["color"]
plt.errorbar(
    gen_err_q,
    train_err_mean,
    yerr=train_err_std,
    label="AMP Training Error",
    color=color,
    linestyle="",
    marker=".",
)
plt.plot(qs, training_error, label="Theoretical Training Error", color=color, linestyle="-")
plt.plot(qs, stability_l1_l2(ms, qs, sigmas, alpha, 1.0, delta_in, delta_out, percentage, beta), label="stability")

plt.xlabel("q")
# plt.yscale("log")
plt.xscale("log")
plt.ylim(-0.5, 4.5)
plt.legend()
plt.grid()

plt.show()
