import numpy as np
from scipy import optimize
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from numba import njit
import cvxpy as cp
from regression_numerics import GTOL_MINIMIZE, MAX_ITER_MINIMIZE, BLEND_GAMP, TOL_GAMP


@njit(error_model="numpy", fastmath=True)
def find_coefficients_L2(ys, xs, reg_param):
    _, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + reg_param * np.identity(d)
    b = np.divide(xs.T.dot(ys), np.sqrt(d))
    return np.linalg.solve(a, b)


def find_coefficients_L1(ys, xs, reg_param):
    _, d = xs.shape
    # w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))
    w = cp.Variable(shape=d)
    obj = cp.Minimize(cp.norm(ys - xs_norm @ w, 1) + 0.5 * reg_param * cp.sum_squares(w))
    prob = cp.Problem(obj)
    prob.solve(eps_abs=1e-3)

    return w.value


@njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    loss = a * np.sum(outliers) - 0.5 * num_outliers * a**2

    non_outliers = linear_loss[~outliers_mask]
    loss += 0.5 * np.dot(non_outliers, non_outliers)
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)
    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_Huber(ys, xs, reg_param, a):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_Huber,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x
