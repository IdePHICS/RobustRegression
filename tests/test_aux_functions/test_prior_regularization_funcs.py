from unittest import TestCase, main
from scipy.optimize import root_scalar
from numpy.random import random
import robust_regression.aux_functions.prior_regularization_funcs as prf


class TestFwL2Regularization(TestCase):
    def test_values(self):
        for _ in range(100):
            gamma = random()
            Lambda = 10 * random() + 0.01
            reg_param = 10 * random() + 0.1

            x_numerics = root_scalar(
                lambda z: 0.5 * reg_param * z**2 + 0.5 * Lambda * (z - gamma / Lambda) ** 2,
            ).root
            self.assertAlmostEqual(prf.f_w_L2_regularization(gamma, Lambda, reg_param), x_numerics)


if __name__ == "__main__":
    main()
