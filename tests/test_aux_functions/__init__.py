from unittest import TestCase
from numpy.random import normal
import inspect


class TestFunction(TestCase):
    def test_values(self, func1, func2):
        n_pars1 = len(inspect.signature(func1).parameters)
        n_pars2 = len(inspect.signature(func2).parameters)
        self.assertEqual(n_pars1, n_pars2)
        n_pars = n_pars1

        for _ in range(100):
            xs = normal(n_pars)
            self.assertAlmostEqual(func1(*xs), func2(*xs))