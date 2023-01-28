import numpy as np

def measure_gen_single(generalization, teacher_vector, xs, delta):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        error_sample = np.sqrt(delta) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        ys = w_xs + error_sample
    return ys


def measure_gen_double(
    generalization, teacher_vector, xs, delta_in, delta_out, percentage
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_in) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_out) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        ys = w_xs + total_error
    return ys


def measure_gen_decorrelated(
    generalization, teacher_vector, xs, delta_in, delta_out, percentage, beta
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_in) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_out) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        factor_in_front = np.where(choice, beta, 1.0)
        ys = factor_in_front * w_xs + total_error
    return ys


def data_generation(
    measure_fun, n_features, n_samples, n_generalization, measure_fun_args
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))

    ys = measure_fun(False, theta_0_teacher, xs, *measure_fun_args)
    ys_gen = measure_fun(True, theta_0_teacher, xs_gen, *measure_fun_args)

    return xs, ys, xs_gen, ys_gen, theta_0_teacher