import math
from time import time
from typing import Optional


def _smoothing_factor(t_e, cutoff):
    """
    Calculates the smoothing factor used in OneEuro filter.

    :param t_e: Time elapsed since last measurement.
    :param cutoff: Minimum cutoff frequency.

    :return: Smoothing factor.
    """
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a, x, x_prev):
    """
    Applies exponential smoothing to a signal.

    :param a: Smoothing factor.
    :param x: New measurement value.
    :param x_prev: Previous measurement value.

    :return: Smoothed measurement value.
    """
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """
    A class to implement the OneEuro filter. This filter is used in many control systems,
    especially those involving velocity sensors or accelerometers.

    https://github.com/casiez/OneEuroFilter
    """

    def __init__(self, x0: float, t0: Optional[float] = None, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Initializes the OneEuro filter.

        :param x0: Initial measurement value.
        :param t0: Initial time value. If None, uses current time.
        :param dx0: Initial derivative of the signal.
        :param min_cutoff: Minimum cutoff frequency.
        :param beta: Parameter used in the cutoff calculation.
        :param d_cutoff: Minimum derivative cutoff frequency.
        """
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = time() if t0 is None else t0

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """
        Computes the filtered signal.

        :param x: New measurement value.
        :param t: Time of new measurement. If None, uses current time.

        :return: Filtered measurement value.
        """
        if t is None:
            t = time()

        # The time elapsed since last measurement.
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
