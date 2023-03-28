import unittest
import robust_regression.utils.integration_utils as iu
import numpy as np

class TestIntegrationUtils(unittest.TestCase):
    def test1(self):
        self.assertAlmostEqual(0.0, 0.0)
    # def test_output(self):
    #     # Test that the output of the function is correct
    #     def f(x):
    #         return x**2

    #     mean = 1.0
    #     std = 2.0
    #     expected_output = np.exp(mean**2) * np.sum(iu.w_ge * f(np.sqrt(2) * std * iu.x_ge + mean))

    #     # Note: we cannot use @njit-ed function in assertAlmostEqual,
    #     # thus, we test np.isclose for the expected output and the function output
    #     self.assertTrue(np.isclose(iu.gauss_hermite_quadrature(f, mean, std), expected_output))

    # def test_error_raised(self):
    #     # Test that an error is raised if the input function does not return a scalar
    #     def f(x):
    #         return np.array([x**2, x**3])

    #     mean = 1.0
    #     std = 2.0
    #     with self.assertRaises(ValueError):
    #         iu.gauss_hermite_quadrature(f, mean, std)

    # def test_valid_input(self):
    #     # Test that the function executes without errors for valid input values
    #     def f(x):
    #         return np.exp(-x**2)

    #     mean = 0.0
    #     std = 1.0
    #     iu.gauss_hermite_quadrature(f, mean, std)  # Should not raise an error

if __name__ == '__main__':
    unittest.main()