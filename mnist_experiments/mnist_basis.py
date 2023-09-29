from robust_regression.regression_numerics.data_generation import measure_gen_decorrelated
from robust_regression.regression_numerics.erm_solvers import _loss_and_gradient_Huber
from scipy import optimize
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10

import numpy as np
import pandas as pd

MAX_ITER_MINIMIZE = 1500
GTOL_MINIMIZE = 1e-8

def main_fixed_instance():
    # np.random.seed(42)

    lambd = -1.5
    a = 10.0
    alpha = 3.00e+01
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300
    d = 2*p

    params = (delta_small, delta_large, percentage, beta)

    # Random feacures from MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], -1)).astype('float')

    W = np.random.randn(784, p) / 10

    n_samples=max(int(np.around(d * alpha)), 1)
    X_rand_test = np.exp(-1j * train_X[:n_samples]@W)
    X_rand_test = np.concatenate((np.real(X_rand_test), np.imag(X_rand_test)), axis=1)


    xs = X_rand_test

    # The rest is the same as in standard plots
    measure_fun = measure_gen_decorrelated
    n_features=d
    n_samples=max(int(np.around(d * alpha)), 1)
    measure_fun_args=params

    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))


    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)

    ground_truth_theta = theta_0_teacher



    
    lambda_list_sim = np.linspace(3, -10, 60)
    E_list = np.zeros_like(lambda_list_sim)
    scale = .1
        
    for i,lambd in enumerate(lambda_list_sim):
        if i == 0:
            w = np.random.normal(loc=0.0, scale=scale, size=(d,))
        else:
            w = estimated_theta
        xs_norm = np.divide(xs, np.sqrt(d))

        bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
        bounds[-1][0] = np.finfo(np.float64).eps * 10

        opt_res = optimize.minimize(
            _loss_and_gradient_Huber,
            w,
            method="L-BFGS-B",
            jac=True,
            args=(xs_norm, ys, lambd, a),
            options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
            bounds=bounds,
        )

        if opt_res.status == 2:
            print(
                "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
                % opt_res.message
            )
            
        estimated_theta = opt_res.x
        E_list[i] = np.sum(np.square(ground_truth_theta - estimated_theta)) / d
        
        print(f"Lambda: {lambd}, E (sim): {E_list[i]}")

    plt.plot(lambda_list_sim, E_list, label=f"Converged simulations (init_norm = {scale})", linestyle="", marker=".")

    # pd.DataFrame(data={"Lambda":lambda_list_sim, "E_simulation":E_list}).to_csv(f"ultimate_plot_a_{a}.csv")

    plt.title(f"Random Foureir features, D={32**2}, {p} Complex features, a={a}, alpha={alpha} (rest as figure 1)")


    plt.xlabel("Lambda")
    plt.ylabel("Generalisation error")
    plt.show()



def solve_ploblem_fixed_a_lambda(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1, start=True, estimated_theta=None):
    d = 2*p

    params = (delta_small, delta_large, percentage, beta)

    # Random feacures from MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], -1)).astype('float')

    W = np.random.randn(784, p) / 10

    n_samples=max(int(np.around(d * alpha)), 1)
    X_rand_test = np.exp(-1j * train_X[:n_samples]@W)
    X_rand_test = np.concatenate((np.real(X_rand_test), np.imag(X_rand_test)), axis=1)


    xs = X_rand_test

    # The rest is the same as in standard plots
    measure_fun = measure_gen_decorrelated
    n_features=d
    n_samples=max(int(np.around(d * alpha)), 1)
    measure_fun_args=params

    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))


    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)

    ground_truth_theta = theta_0_teacher


        
    if start == True:
        w = np.random.normal(loc=0.0, scale=scale, size=(d,))
    else:
        w = estimated_theta
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_Huber,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, lambd, a),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        print(
            "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )
        
    estimated_theta = opt_res.x
    E_estimation = np.sum(np.square(ground_truth_theta - estimated_theta)) / d


    return E_estimation, estimated_theta

def single_problem_main():
    lambd = .1
    a = 10.0
    alpha = 3.00e+01
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    E_estimation, estimated_thteta = solve_ploblem_fixed_a_lambda(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1, start=True, estimated_theta=None)
    print(f"Lambda: {lambd}, E (sim): {E_estimation}")


def solve_ploblem_fixed_a(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1):
    d = 2*p

    params = (delta_small, delta_large, percentage, beta)

    # Random feacures from MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], -1)).astype('float')

    W = np.random.randn(784, p) / 10

    n_samples=max(int(np.around(d * alpha)), 1)
    X_rand_test = np.exp(-1j * train_X[:n_samples]@W)
    X_rand_test = np.concatenate((np.real(X_rand_test), np.imag(X_rand_test)), axis=1)


    xs = X_rand_test

    # The rest is the same as in standard plots
    measure_fun = measure_gen_decorrelated
    n_features=d
    n_samples=max(int(np.around(d * alpha)), 1)
    measure_fun_args=params

    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))


    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)

    ground_truth_theta = theta_0_teacher


        
    w = np.random.normal(loc=0.0, scale=scale, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    def minimize_fun(lambd):
        opt_res = optimize.minimize(
            _loss_and_gradient_Huber,
            w,
            method="L-BFGS-B",
            jac=True,
            args=(xs_norm, ys, lambd, a),
            options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
            bounds=bounds,
        )

        if opt_res.status == 2:
            print(
                "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
                % opt_res.message
            )
            
        estimated_theta = opt_res.x
        E_estimation = np.sum(np.square(ground_truth_theta - estimated_theta)) / d
        return E_estimation

    obj = optimize.minimize(
        minimize_fun,
        x0=lambd,
        bounds=[(1e-8,None)],
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "adaptive": True,
        },
    )

    E_estimation = obj.fun
    lambda_opt = obj.x

    return E_estimation, lambda_opt




def sweep_alpha_main_fixed_a_lambda():
    lambd = 1.0
    a = 10.0
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    alpha_list = np.logspace(-1,2,16)
    E_estimation_list = np.zeros(len(alpha_list))
    for i,alpha in enumerate(alpha_list):
        if i == 0:
            E_estimation_list[i], estimated_thteta = solve_ploblem_fixed_a_lambda(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1, start=True, estimated_theta=None)
        else:
            E_estimation_list[i], estimated_thteta = solve_ploblem_fixed_a_lambda(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1, start=False, estimated_theta=estimated_thteta)
        print(f"Alpha: {alpha}, E (sim): {E_estimation_list[i]}")
    
    plt.plot(alpha_list, E_estimation_list, linestyle="", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Estimation error")
    plt.show()


def solve_ploblem(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1):
    d = 2*p

    params = (delta_small, delta_large, percentage, beta)

    # Random feacures from MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], -1)).astype('float')

    W = np.random.randn(784, p) / 10

    n_samples=max(int(np.around(d * alpha)), 1)
    X_rand_test = np.exp(-1j * train_X[:n_samples]@W)
    X_rand_test = np.concatenate((np.real(X_rand_test), np.imag(X_rand_test)), axis=1)


    xs = X_rand_test

    # The rest is the same as in standard plots
    measure_fun = measure_gen_decorrelated
    n_features=d
    n_samples=max(int(np.around(d * alpha)), 1)
    measure_fun_args=params

    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))


    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)

    ground_truth_theta = theta_0_teacher


        
    w = np.random.normal(loc=0.0, scale=scale, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    def minimize_fun(params):
        lambd, a = params
        opt_res = optimize.minimize(
            _loss_and_gradient_Huber,
            w,
            method="L-BFGS-B",
            jac=True,
            args=(xs_norm, ys, lambd, a),
            options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
            bounds=bounds,
        )

        if opt_res.status == 2:
            print(
                "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
                % opt_res.message
            )
            
        estimated_theta = opt_res.x
        E_estimation = np.sum(np.square(ground_truth_theta - estimated_theta)) / d
        return E_estimation

    obj = optimize.minimize(
        minimize_fun,
        x0=(lambd,a),
        bounds=[(-1e-8,None),(1e-8,None)],
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "adaptive": True,
        },
    )

    E_estimation = obj.fun
    lambda_opt, a_opt = obj.x

    return E_estimation, lambda_opt, a_opt


def sweep_alpha_main_fixed_a():
    lambd = .1
    a = 1e-3
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0
    repetitions = 16
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    alpha_list = np.logspace(-1,3,32)
    E_estimation_list = np.zeros((repetitions, len(alpha_list)))
    lambda_list = np.ones((repetitions, len(alpha_list))) * lambd
    for r in range(repetitions):
        for i,alpha in enumerate(alpha_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a, lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"Alpha: {alpha}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}")
    
    pd.DataFrame({"alpha": alpha_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0)}).to_csv("review_alpha_sweep_repetitions.csv")

    plt.plot(alpha_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Estimation error")
    plt.show()

def sweep_alpha_main():
    lambd = .1
    a = 1
    delta_small = 1
    delta_large = 5
    percentage = .3
    beta = 0
    repetitions = 2
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    alpha_list = np.logspace(-1,3,32)
    E_estimation_list = np.zeros((repetitions, len(alpha_list)))
    lambda_list = np.ones((repetitions, len(alpha_list))) * lambd
    a_list = np.ones((repetitions, len(alpha_list))) * a
    for r in range(repetitions):
        for i,alpha in enumerate(alpha_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a_list[r,i-1], lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"Alpha: {alpha}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}, a: {a_list[r,i]}")
    
    pd.DataFrame({"alpha": alpha_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0), "a_mean": a_list.mean(axis=0), "a_std": a_list.std(axis=0)}).to_csv("review_alpha_sweep_repetitions_opt.csv")

    plt.plot(alpha_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Estimation error")
    plt.show()


def sweep_epsilon_main():
    lambd = .1
    a = 1
    alpha = 10
    delta_small = 1
    delta_large = 5
    beta = 0
    repetitions = 16
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    percentage_list = np.linspace(0,1,32)
    E_estimation_list = np.zeros((repetitions, len(percentage_list)))
    lambda_list = np.ones((repetitions, len(percentage_list))) * lambd
    a_list = np.ones((repetitions, len(percentage_list))) * a
    for r in range(repetitions):
        for i,percentage in enumerate(percentage_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a_list[r,i-1], lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"Percentage: {percentage}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}, a: {a_list[r,i]}")
    
    pd.DataFrame({"Percentage": percentage_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0), "a_mean": a_list.mean(axis=0), "a_std": a_list.std(axis=0)}).to_csv("review_epsilon_sweep_repetitions_opt.csv")

    plt.plot(percentage_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.yscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("Estimation error")
    plt.show()

def sweep_epsilon_main_fixed_a():
    lambd = .1
    a = 100
    alpha = 10
    delta_small = 1
    delta_large = 5
    beta = 0
    repetitions = 16
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    percentage_list = np.linspace(0,1,32)
    E_estimation_list = np.zeros((repetitions, len(percentage_list)))
    lambda_list = np.ones((repetitions, len(percentage_list))) * lambd
    a_list = np.ones((repetitions, len(percentage_list))) * a
    for r in range(repetitions):
        for i,percentage in enumerate(percentage_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a_list[r,i-1], lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"Percentage: {percentage}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}")
    
    pd.DataFrame({"Percentage": percentage_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0)}).to_csv(f"review_epsilon_sweep_repetitions_a_{a}_short.csv")

    plt.plot(percentage_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.yscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("Estimation error")
    plt.show()


def sweep_delta_main():
    lambd = .1
    a = 1
    alpha = 10
    delta_small = 1
    beta = 0
    repetitions = 16
    percentage = 0.3
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    delta_list = np.logspace(1,4,36)[1:13]
    E_estimation_list = np.zeros((repetitions, len(delta_list)))
    lambda_list = np.ones((repetitions, len(delta_list))) * lambd
    a_list = np.ones((repetitions, len(delta_list))) * a
    for r in range(repetitions):
        for i,delta_large in enumerate(delta_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i], a_list[r,i] = solve_ploblem(a_list[r,i-1], lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"D_out: {delta_large}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}, a: {a_list[r,i]}")
            
    pd.DataFrame({"Delta": delta_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0), "a_mean": a_list.mean(axis=0), "a_std": a_list.std(axis=0)}).to_csv(f"review_delta_sweep_repetitions_short.csv")

    plt.plot(delta_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.yscale("log")
    plt.xlabel("delta")
    plt.ylabel("Estimation error")
    plt.show()

def sweep_delta_main_fixed_a():
    lambd = .1
    a = 100
    alpha = 10
    delta_small = 1
    beta = 0
    repetitions = 16
    percentage = 0.3
    
    # Number of features (it will double because we take immaginary and real components so d = 2*p)
    p = 300

    delta_list = np.logspace(1,4,36)[1:13]
    E_estimation_list = np.zeros((repetitions, len(delta_list)))
    lambda_list = np.ones((repetitions, len(delta_list))) * lambd
    a_list = np.ones((repetitions, len(delta_list))) * a
    for r in range(repetitions):
        for i,delta_large in enumerate(delta_list):
            if i == 0:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a, lambd, alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            else:
                E_estimation_list[r,i], lambda_list[r,i] = solve_ploblem_fixed_a(a_list[r,i-1], lambda_list[r,i-1], alpha, p, delta_small, delta_large, percentage, beta, scale=.1)
            print(f"D_out: {delta_large}, E (sim): {E_estimation_list[r,i]}, lambda: {lambda_list[r,i]}")
    
    pd.DataFrame({"Delta": delta_list, "E_mean": E_estimation_list.mean(axis=0), "E_std": E_estimation_list.std(axis=0), "lambda_mean": lambda_list.mean(axis=0), "lambda_std": lambda_list.std(axis=0)}).to_csv(f"review_delta_sweep_repetitions_fixed_a_short.csv")

    plt.plot(delta_list, E_estimation_list.mean(axis=0), linestyle="", marker=".")
    plt.yscale("log")
    plt.xlabel("delta")
    plt.ylabel("Estimation error")
    plt.show()


if __name__=="__main__":
    sweep_delta_main()
