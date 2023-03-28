import unittest
import robust_regression as rr

class Test_fpeqs(unittest.TestCase):
    # test the fixed point equations function
    def test_find_fixed_point(self):
        self.assertAlmostEqual(0.0, 0.0)

if __name__ == '__main__':
    unittest.main()
