from unittest import TestCase, main
from numpy import logspace, empty
import robust_regression.sweeps.alpha_sweeps as alsw

class TestSweepAlphaFixedPoint(TestCase):
    def test_alpha_wrong_order(self):
        alpha_min = 10.0
        alpha_max = 1.0
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(lambda x: x, lambda x : x, alpha_min, alpha_max, 100, {}, {}, (1.0,2.0,3.0), [], [])

    def test_alpha_negative(self):
        alpha_min = -1.0
        alpha_max = 1.0
        with self.assertRaises(ValueError):
            alsw.sweep_alpha_fixed_point(lambda x: x, lambda x : x, alpha_min, alpha_max, 100, {}, {}, (1.0,2.0,3.0), [], [])
    

if __name__ == '__main__':
    main()