from unittest import TestCase
import numpy as np
import random
import inspect
import numba


class FunctionComparisonTest(TestCase):
    def get_num_args(self, func):
        if isinstance(func, numba.core.registry.CPUDispatcher):
            py_func = func.py_func
            return len(inspect.signature(py_func).parameters)
        elif isinstance(func, numba.np.ufunc.dufunc.DUFunc):
            return func.nin
        else:
            return len(inspect.signature(func).parameters)

    def sample_arg(self, signature):
        if signature == "+":
            return random.uniform(0, 100)
        elif signature == "u":
            return random.uniform(-100, 100)
        else:
            raise ValueError("Invalid argument signature")

    def compare_two_functions(
        self, func1, func2, num_points=100, tolerance=1e-4, arg_signatures=None
    ):
        num_args_func1 = self.get_num_args(func1)
        num_args_func2 = self.get_num_args(func2)

        if num_args_func1 != num_args_func2:
            raise ValueError("Functions have different number of arguments")

        if arg_signatures is not None and len(arg_signatures) != num_args_func1:
            raise ValueError("Length of arg_signatures doesn't match the number of arguments")

        for _ in range(num_points):
            if arg_signatures is None:
                args = [np.array([random.uniform(-100, 100)]) for _ in range(num_args_func1)]
            else:
                args = [np.array([self.sample_arg(sig)]) for sig in arg_signatures]

            result1 = func1(*args)
            result2 = func2(*args)

            # Convert results to scalar values for comparison
            result1_scalar = result1.item() if isinstance(result1, np.ndarray) else result1
            result2_scalar = result2.item() if isinstance(result2, np.ndarray) else result2

            self.assertAlmostEqual(
                result1_scalar,
                result2_scalar,
                delta=tolerance,
                msg=f"Functions outputs don't match at args = {args}: {result1_scalar} != {result2_scalar}",
            )