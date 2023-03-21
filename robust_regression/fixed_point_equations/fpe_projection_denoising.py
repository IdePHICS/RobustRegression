from numba import njit
from math import sqrt

# @njit(error_model="numpy", fastmath=True)
def var_func_projection_denoising(m_hat, q_hat, sigma_hat, q_fixed):
    eta_hat = m_hat **2 / q_hat
    m = m_hat * sqrt(q_fixed) / sqrt(m_hat**2 + q_hat) # sqrt((q_hat + m_hat**2) * q_fixed / q_hat**3)  # (1 + eta_hat) * sqrt(q_fixed) * m_hat / q_hat
    q = q_fixed # (1 + eta_hat) * q_fixed
    sigma = sqrt(q_fixed * q_hat / (q_hat + m_hat**2))
    return m, q, sigma