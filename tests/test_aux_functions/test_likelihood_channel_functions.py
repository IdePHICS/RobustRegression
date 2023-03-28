from unittest import TestCase, main
from scipy.optimize import root_scalar
from numpy.random import random, normal
from math import sqrt
import robust_regression.aux_functions.likelihood_channel_functions as lcf


class TestFoutL2(TestCase):
    def test_values(self):
        for _ in range(100):
            y = sqrt(10) * normal()
            V = 10 * random() + 0.01
            omega = 10 * normal()
            x_numerics = root_scalar(
                lambda z: 0.5 * (y - z) ** 2 + 1.0 / (2.0 * V) * (z - omega) ** 2,
            ).root
            self.assertAlmostEqual(lcf.f_out_L2(y, omega, V), x_numerics)


class TestFoutL1(TestCase):
    def test_values(self):
        for _ in range(100):
            y = sqrt(10) * normal()
            V = 10 * random() + 0.01
            omega = 10 * normal()
            x_numerics = root_scalar(
                lambda z: abs(y - z) + 1.0 / (2.0 * V) * (z - omega) ** 2,
            ).root
            self.assertAlmostEqual(lcf.f_out_L1(y, omega, V), x_numerics)


class TestFoutHuber(TestCase):
    def test_values(self):
        for _ in range(100):
            y = sqrt(10) * normal()
            V = 10 * random() + 0.01
            omega = 10 * normal()
            a = 10 * random() + 0.01

            def huber_loss(x,y,a):
                if abs(x-y) <= a:
                    return 0.5 * (x-y)**2
                else:
                    return a * abs(x-y) - 0.5 * a**2
                
            x_numerics = root_scalar(
                lambda z: huber_loss(y,z,a) + 1.0 / (2.0 * V) * (z - omega) ** 2,
            ).root
            self.assertAlmostEqual(lcf.f_out_Huber(y, omega, V, a), x_numerics)


if __name__ == "__main__":
    main()
