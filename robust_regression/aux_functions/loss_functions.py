from numba import vectorize

@vectorize("float64(float64, float64)")
def l2_loss(y: float, z: float):
    return 0.5 * (y - z) ** 2


@vectorize("float64(float64, float64)")
def l1_loss(y: float, z: float):
    return abs(y - z)


@vectorize("float64(float64, float64, float64)")
def huber_loss(y: float, z: float, a: float):
    if abs(y - z) < a:
        return 0.5 * (y - z) ** 2
    else:
        return a * abs(y - z) - 0.5 * a**2