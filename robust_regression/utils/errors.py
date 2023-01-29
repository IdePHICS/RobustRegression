class ConvergenceError(ValueError):
    def __init__(self, fname, n_iteration, *args, **kwargs):
        super().__init__(
            "The function {} didn't converge after {:d} iterations".format(fname, n_iteration),
            *args,
            **kwargs
        )
