import unittest
import robust_regression as rr

class Test_fpeqs(unittest.TestCase):
    # test the fixed point equations function
    def test_find_fixed_point(self):
        def var(x, m, q, sigma):
            return m + q * x + sigma * x ** 2

        def var_hat(x, m, q, sigma, eps):
            return var(x, m, q, sigma) + eps

        m, q, sigma = 0.6, 0.01, 0.9
        eps_min, eps_max, n_eps_pts = 1e-2, 1e2, 10
        epsilons = rr.aux_functions.misc.logspace(
            rr.aux_functions.misc.log10(eps_min),
            rr.aux_functions.misc.log10(eps_max),
            n_eps_pts,
        )
        out_list = [rr.np.empty(n_eps_pts) for _ in range(1)]
        ms_qs_sigmas = rr.np.empty((n_eps_pts, 3))

        reg_param = 1e-3
        var_hat_args = (m, q, sigma)
        initial_cond = (0.6, 0.01, 0.9)

        for idx, eps in enumerate(epsilons):
            current_var_hat_args = list()
            for jdx, arg in enumerate(var_hat_args):
                if jdx != 2:
                    current_var_hat_args.append(arg)
                else:
                    current_var_hat_args.append(eps)
            current_var_hat_args = tuple(current_var_hat_args)

            ms_qs_sigmas[idx] = rr.fixed_point_equations.fpeqs.find_fixed_point(
                var, var_hat, reg_param, initial_cond, current_var_hat_args
            )
            old_initial_cond = tuple(ms_qs_sigmas[idx])

        self.assertAlmostEqual(ms_qs_sigmas[0][0], 0.6000000000000001)
        self.assertAlmostEqual(ms_qs_sigmas[0][1], 0.01)
        self.assertAlmostEqual(ms_qs_sigmas[0][2], 0.9)

        self.assertAlmostEqual(ms_qs_sigmas[1][0], 0.6000000000000001)
        self.assertAlmostEqual(ms_qs_sigmas[1][1], 0.01)
        self.assertAlmostEqual(ms_qs_sigmas[1][2], 0.9)

        self.assertAlmostEqual(ms_qs_sigmas[2][0], 0.6000000000000001)
        self.assertAlmostEqual(ms_qs_sigmas[2][1], 0.01)

if __name__ == '__main__':
    unittest.main()
