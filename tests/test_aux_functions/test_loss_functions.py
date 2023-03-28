from unittest import TestCase, main
from numpy.random import randn, normal
import robust_regression.aux_functions.loss_functions as lf


class TestL2LossFunctions(TestCase):
    def test_values(self):
        def correct_loss(x, y):
            return 0.5 * (x - y) ** 2

        for _ in range(100):
            x = randn()
            y = randn()
            self.assertAlmostEqual(lf.l2_loss(x, y), correct_loss(x, y))

        for _ in range(100):
            x = normal()
            y = normal()
            self.assertAlmostEqual(lf.l2_loss(x, y), correct_loss(x, y))


class TestL1LossFunctions(TestCase):
    def test_values(self):
        def correct_loss(x, y):
            return abs(x - y)

        for _ in range(100):
            x = randn()
            y = randn()
            self.assertAlmostEqual(lf.l1_loss(x, y), correct_loss(x, y))

        for _ in range(100):
            x = normal()
            y = normal()
            self.assertAlmostEqual(lf.l1_loss(x, y), correct_loss(x, y))


class TestHuberLossFunctions(TestCase):
    def test_values(self):
        def correct_loss(x, y, a):
            if abs(x - y) < a:
                return 0.5 * (x - y) ** 2
            else:
                return a * abs(x - y) - 0.5 * a**2

        for _ in range(100):
            x = randn()
            y = randn()
            a = randn()
            self.assertAlmostEqual(lf.huber_loss(x, y, a), correct_loss(x, y, a))

        for _ in range(100):
            x = normal()
            y = normal()
            a = normal()
            self.assertAlmostEqual(lf.huber_loss(x, y, a), correct_loss(x, y, a))


if __name__ == "__main__":
    main()
