import robust_regression.regression_numerics.amp_funcs as amp
import robust_regression.regression_numerics.data_generation as dg
import robust_regression.aux_functions.prior_regularization_funcs as priors
import robust_regression.aux_functions.likelihood_channel_functions as like
import numpy as np
import matplotlib.pyplot as plt
import robust_regression.fixed_point_equations as fpe
from robust_regression.utils.errors import ConvergenceError

print("here")

f_w_args = (1.0)
f_out_args = (1.0, 5.0, 0.3, 0.0)

n_features = 1000
repetitions = 5

alpha = 5.0

gen_err_param = list()
gen_err_mean = list()
gen_err_std = list()

reg_params = np.linspace(0.01, -1.6, 40)

for idx, reg_param in enumerate(reg_params):
    print(idx, reg_param)

    all_gen_errors = np.empty(repetitions)

    try:
        for jdx in range(repetitions):
            print("\tjdx ", jdx)
            xs, ys, _, _, theta_0_teacher = dg.data_generation(
                dg.measure_gen_decorrelated, 
                n_features, 
                max(int(np.around(n_features * alpha)), 1), 
                1, 
                (1.0, 5.0, 0.3, 0.0)
            )

            f_w_args_l2_reg = (float(reg_param),)
            f_out_args_l2 = list()

            estimated_theta = amp.GAMP_algorithm_unsimplified(
                priors.f_w_L2_regularization,
                priors.Df_w_L2_regularization,
                like.f_out_L2,
                like.Df_out_L2,
                ys,
                xs,
                f_w_args_l2_reg,
                f_out_args_l2,
                theta_0_teacher,
                blend=0.2,
            )

            all_gen_errors[jdx] = np.divide(
                np.sum(np.square(theta_0_teacher - estimated_theta)), n_features
            )

            del xs
            del ys
            del theta_0_teacher
            
        gen_err_param.append(float(reg_param))
        gen_err_mean.append(np.mean(all_gen_errors))
        gen_err_std.append(np.std(all_gen_errors))
    except ConvergenceError as e:
        print(e)
        break
        gen_err_mean.append(np.nan)
        gen_err_std.append(np.nan)


print(gen_err_mean)
plt.errorbar(gen_err_param, gen_err_mean, yerr=gen_err_std, label="AMP")

plt.ylabel("E_gen")
plt.yscale('log')
plt.grid()

plt.show()