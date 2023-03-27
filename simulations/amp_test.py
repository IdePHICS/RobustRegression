import robust_regression.regression_numerics.amp_funcs as amp
import robust_regression.regression_numerics.data_generation as dg
import robust_regression.aux_functions.prior_regularization_funcs as priors
import robust_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
import matplotlib.pyplot as plt
import robust_regression.fixed_point_equations as fpe
from robust_regression.utils.errors import ConvergenceError

# generate the data
# xs, ys, _, _, theta_0_teacher = dg.data_generation(
#     dg.measure_gen_decorrelated, 
#     500, 
#     50000, 
#     1, 
#     (0.1, 5.0, 0.1, 0.0)
# )

print("here")

f_w_args = (1.0)
f_out_args = (1.0, 5.0, 0.3, 0.0)

# print(
#     amp.GAMP_algorithm(
#         priors.f_w_Bayes_gaussian_prior,
#         priors.Df_w_Bayes_gaussian_prior,
#         like.f_out_Bayes_decorrelated_noise,
#         like.Df_out_Bayes_decorrelated_noise,
#         ys,
#         xs,
#         f_w_args,
#         f_out_args
#     )
# )

# data_hub = np.genfromtxt(
#     "./data/FIGURE_1_data_uncorrelated_unbounded.csv",
#     delimiter=',',
#     skip_header=1
# )
# alphas_prev = data_hub[:,0]
# err_prev_l2 = data_hub[:,1]
# lambda_prev_l2 = data_hub[:,2]
# err_prev_l1 = data_hub[:,3]
# lambda_prev_l1 = data_hub[:,4]
# err_prev_hub = data_hub[:,5]
# lambda_prev_hub = data_hub[:,6]
# a_prev_hub = data_hub[:,7]

n_features = 500
repetitions = 10

alphas = list()
gen_error_mean_hub = list()
gen_error_std_hub = list()
gen_error_mean_l1 = list()
gen_error_std_l1 = list()
gen_error_mean_l2 = list()
gen_error_std_l2 = list()

data = np.genfromtxt(
    "./simulations/data/TEST_alpha_sweep_L2.csv",
    delimiter=',',
    skip_header=1
)
alphas_prev_l2 = data[:,0]
err_prev_l2 = data[:,1]
lambda_prev_l2 = data[:,2]

data = np.genfromtxt(
    "./simulations/data/TEST_alpha_sweep_L1.csv",
    delimiter=',',
    skip_header=1
)
alphas_prev_l1 = data[:,0]
err_prev_l1 = data[:,1]
lambda_prev_l1 = data[:,2]


data = np.genfromtxt(
    "./simulations/data/TEST_alpha_sweep_Huber.csv",
    delimiter=',',
    skip_header=1
)
alphas_prev_hub = data[:,0]
err_prev_hub = data[:,1]
lambda_prev_hub = data[:,2]
a_prev_hub = data[:,3]

alphas_l2 = list()
alphas_l1 = list()
alphas_hub = list()

# alphas = np.logspace(0, np.log10(100), 25)
# gen_error_mean = np.empty_like(alphas)
# gen_error_std = np.empty_like(alphas)

# for idx, (al, lam_l2, lam_l1, lam_hub, a_hub) in enumerate(zip(alphas_prev, lambda_prev_l2, lambda_prev_l1, lambda_prev_hub, a_prev_hub)):
for idx, (al_l2, lam_l2, al_l1, lam_l1, al_hub, lam_hub, a_hub) in enumerate(zip(alphas_prev_l2, lambda_prev_l2, alphas_prev_l1, lambda_prev_l1, alphas_prev_hub, lambda_prev_hub, a_prev_hub)):
    # if al <= 0.5 or al >= 20 or idx % 20 != 0:
    #     continue

    if idx % 5 != 0:
        continue

    if al_hub > 10:
        continue

    # alphas.append(float(al))

    print(idx, " - ", al_l2, lam_l2, " - ", al_l1, lam_l1, " - ", al_hub, lam_hub, a_hub)

    all_gen_errors = np.empty(repetitions)
    for jdx in range(repetitions):
        print("\t", jdx)
        xs, ys, _, _, theta_0_teacher = dg.data_generation(
            dg.measure_gen_decorrelated, 
            n_features, 
            max(int(np.around(n_features * al_l2)), 1), 
            1, 
            (1.0, 5.0, 0.3, 0.0)
        )

        f_w_args_l2_reg = (float(lam_l2),)
        f_out_args_l2 = list()

        estimated_theta = amp.GAMP_algorithm_unsimplified(
            priors.f_w_L2_regularization,
            priors.f_w_L2_regularization,
            like.f_out_L2,
            like.Df_out_L2,
            ys,
            xs,
            f_w_args_l2_reg,
            f_out_args_l2,
            0.1 * np.random.random(n_features), # theta_0_teacher + 0.0001 * np.random.randn(n_features),
            blend=1.0,
        )

        all_gen_errors[jdx] = np.divide(
            np.sum(np.square(theta_0_teacher - estimated_theta)), n_features
        )

        del xs
        del ys
        del theta_0_teacher
    
    alphas_l2.append(al_hub)
    gen_error_mean_l2.append(np.mean(all_gen_errors))
    gen_error_std_l2.append(np.std(all_gen_errors))

    # ---------
    all_gen_errors = np.empty(repetitions)

    try:
        for jdx in range(repetitions):
            print("\t", jdx)
            xs, ys, _, _, theta_0_teacher = dg.data_generation(
                dg.measure_gen_decorrelated, 
                n_features, 
                max(int(np.around(n_features * al_l1)), 1), 
                1, 
                (1.0, 5.0, 0.3, 0.0)
            )

            f_w_args_l2_reg = (float(lam_l1),)
            f_out_args_l1 = list()

            estimated_theta = amp.GAMP_algorithm_unsimplified(
                priors.f_w_L2_regularization,
                priors.Df_w_L2_regularization,
                like.f_out_L1,
                like.Df_out_L1,
                ys,
                xs,
                f_w_args_l2_reg,
                f_out_args_l1,
                theta_0_teacher, # 0.5 * np.random.random(n_features) + 0.01 # theta_0_teacher
            )

            all_gen_errors[jdx] = np.divide(
                np.sum(np.square(theta_0_teacher - estimated_theta)), n_features
            )

            del xs
            del ys
            del theta_0_teacher
        
        alphas_l1.append(al_hub)
        gen_error_mean_l1.append(np.mean(all_gen_errors))
        gen_error_std_l1.append(np.std(all_gen_errors))
    except ConvergenceError:
        # gen_error_mean_l1.append(np.mean(all_gen_errors))
        # gen_error_std_l1.append(np.std(all_gen_errors))
        aaaa = 1

    # --------------
    all_gen_errors = np.empty(repetitions)
    for jdx in range(repetitions):
        print("\t", jdx)
        xs, ys, _, _, theta_0_teacher = dg.data_generation(
            dg.measure_gen_decorrelated, 
            n_features, 
            max(int(np.around(n_features * al_hub)), 1), 
            1, 
            (1.0, 5.0, 0.3, 0.0)
        )

        # f_w_args_l2_reg = (float(lam_hub),)
        # f_out_args_Huber = (float(a_hub),)
        # print(lam_l1)
        f_w_args_l2_reg = (float(lam_hub),)
        f_out_args_Huber = (float(a_hub),)

        estimated_theta = amp.GAMP_algorithm_unsimplified(
            priors.f_w_L2_regularization,
            priors.Df_w_L2_regularization,
            like.f_out_Huber,
            like.Df_out_Huber,
            ys,
            xs,
            f_w_args_l2_reg,
            f_out_args_Huber,
            0.5 * np.random.random(n_features) + 0.01 # theta_0_teacher
        )

        all_gen_errors[jdx] = np.divide(
            np.sum(np.square(theta_0_teacher - estimated_theta)), n_features
        )

        del xs
        del ys
        del theta_0_teacher
    
    alphas_hub.append(al_hub)
    gen_error_mean_hub.append(np.mean(all_gen_errors))
    gen_error_std_hub.append(np.std(all_gen_errors))

    # gen_error_mean[idx], gen_error_std[idx] = np.mean(all_gen_errors), np.std(all_gen_errors)

    # if al > 270:
    #     break


# np.savetxt(
#     "./data/AMP_BO_huber_bounded_random_init.csv",
#     np.vstack((np.array(alphas), np.array(gen_error_mean_l2), np.array(gen_error_std_l2), np.array(gen_error_mean_l1), np.array(gen_error_std_l1) ,np.array(gen_error_mean_hub), np.array(gen_error_std_hub))).T,
#     delimiter=",",
#     header="alphas, err_mean_l2, err_std_l2, err_mean_l1, err_std_l1, err_mean_hub, err_std_hub",
# )

# np.savetxt(
#     "./data/AMP_BO_huber_bounded_random_init.csv",
#     np.vstack((np.array(alphas), np.array(gen_error_mean_l1), np.array(gen_error_std_l1))).T,
#     delimiter=",",
#     header="alphas, err_mean_l1, err_std_l1",
# )

# np.savetxt(
#     "./data/AMP_BO_huber_unbounded_random_init.csv",
#     np.vstack((np.array(alphas), np.array(gen_error_mean_l2), np.array(gen_error_std_l2),np.array(gen_error_mean_hub), np.array(gen_error_std_hub))).T,
#     delimiter=",",
#     header="# alphas, err_mean, err_std",
# )

# data = np.genfromtxt(
#    "./data/AMP_BO.csv",
#     delimiter=",",
#     skip_header=1
# )
# alphas = data[:,0]
# gen_error_mean = data[:,1]
# gen_error_std = data[:,2]


# import src.fpeqs as fp
# from src.fpeqs_BO import (
#     var_func_BO,
#     var_hat_func_BO_single_noise,
#     var_hat_func_BO_num_double_noise,
#     var_hat_func_BO_num_decorrelated_noise,
# )

# while True:
#     m = 0.89 * np.random.random() + 0.1
#     q = 0.89 * np.random.random() + 0.1
#     sigma = 0.89 * np.random.random() + 0.1
#     if np.square(m) < q + 0.1 * q and np.square(m) < q + 5.0 * q:
#         initial_condition = [m, q, sigma]
#         break

# pap = {
#     "delta_small": 0.1,
#     "delta_large": 5.0,
#     "percentage": 0.1,
#     "beta": 0.0,
# }

# alphas_BO, (errors_BO,) = fp.no_parallel_different_alpha_observables_fpeqs(
#     var_func_BO,
#     var_hat_func_BO_num_decorrelated_noise,
#     alpha_1=0.1,
#     alpha_2=1000,
#     n_alpha_points=70,
#     initial_cond=initial_condition,
#     var_hat_kwargs=pap,
# )

# plt.subplot(211)
plt.figure(figsize=(10, 8))

print(len(alphas_l2), len(gen_error_mean_l2), len(gen_error_std_l2))
print(len(alphas_l1), len(gen_error_mean_l1), len(gen_error_std_l1))
print(len(alphas_hub), len(gen_error_mean_hub), len(gen_error_std_hub))

plt.plot(alphas_prev_l2, err_prev_l2, label="L2")
plt.errorbar(alphas_l2, gen_error_mean_l2, yerr=gen_error_std_l2, label="AMP L2")

plt.plot(alphas_prev_l1, err_prev_l1, label="L1")
plt.errorbar(alphas_l1, gen_error_mean_l1, yerr=gen_error_std_l1, label="AMP L1")

plt.plot(alphas_prev_hub, err_prev_hub, label="Huber")
plt.errorbar(alphas_hub, gen_error_mean_hub, yerr=gen_error_std_hub, label="AMP Hub")

plt.ylabel("E_gen")
plt.xscale('log')
plt.yscale('log')
# plt.xlim([0.1, 100])
# plt.ylim([1e-1, 1.5])
plt.legend()
plt.grid()

plt.show()