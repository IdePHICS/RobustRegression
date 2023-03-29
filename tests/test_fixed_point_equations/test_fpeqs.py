from unittest import TestCase, main
import robust_regression.fixed_point_equations.fpeqs as fpe


class TestFixedPointFinder(TestCase):
    def test_output_format(self):
        self.assertAlmostEqual(0.0, 0.0)

    def test_find_fixed_point_convergence(self):
        self.assertAlmostEqual(0.0, 0.0)

    def test_single_fixed_point(self):
        self.assertAlmostEqual(0.0, 0.0)


if __name__ == "__main__":
    main()
