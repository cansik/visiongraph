"""
Source: https://github.com/HoBeom/OneEuroFilter-Numpy
MIT License
"""
from time import time
from typing import Optional

import numpy as np

from visiongraph.dsp.BaseFilterNumpy import BaseFilterNumpy

EPS = 1e-7


def _smoothing_factor(t_e, cutoff):
    """
    Compute the smoothing factor.

    :param t_e: The time difference.
    :param cutoff: The cutoff frequency.

    :return: The smoothing factor.
    """
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a, x, x_prev):
    """
    Compute the exponential smoothing of a signal.

    :param a: The smoothing factor.
    :param x: The input signal.
    :param x_prev: The previous value of the signal.

    :return: The smoothed signal.
    """
    return a * x + (1 - a) * x_prev


class OneEuroFilterNumpy(BaseFilterNumpy):
    """
    A class to implement the OneEuro filter, which is used for filtering
    time-series data. It combines an exponential smoothing with a least-squares
    estimation of the acceleration signal.
    """

    def __init__(self, x0: np.ndarray, t0: Optional[float] = None, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0,
                 invalid_value: Optional[float] = None):
        """
        Initialize the OneEuro filter.

        :param x0: The initial signal.
        :param t0: The initial time. Defaults to None.
        :param dx0: The initial derivative of the signal. Defaults to 0.0.
        :param min_cutoff: The minimum cutoff frequency. Defaults to 1.0.
        :param beta: The smoothing factor for the acceleration. Defaults to 0.0.
        :param d_cutoff: The cutoff frequency for the derivative. Defaults to 1.0.
        :param invalid_value: The value that indicates an invalid data point. Defaults to None.
        """
        # The parameters.
        self.data_shape = x0.shape
        self.min_cutoff = np.full(x0.shape, min_cutoff)
        self.beta = np.full(x0.shape, beta)
        self.d_cutoff = np.full(x0.shape, d_cutoff)

        # Previous values.
        self.x_prev = x0.astype(float)
        self.dx_prev = np.full(x0.shape, dx0)
        self.t_prev = time() if t0 is None else t0

        self.invalid_value = invalid_value

    def re_init(self, x: np.ndarray):
        """
        Re-initialize the filter with a new input signal.

        :param x: The new input signal.
        """
        self.data_shape = x.shape
        self.min_cutoff = np.full(self.data_shape, self.min_cutoff.flat[0])
        self.beta = np.full(self.data_shape, self.beta.flat[0])
        self.d_cutoff = np.full(self.data_shape, self.d_cutoff.flat[0])

        self.x_prev = x.astype(float)
        self.dx_prev = np.full(self.data_shape, 0.0)  # reset dx_prev

        self.t_prev = time()

    def __call__(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Compute the filtered signal.

        :param x: The input signal.
        :param t: The current time. Defaults to None.

        :return: The filtered signal.
        """
        if x.shape != self.data_shape:
            self.re_init(x)
            return x

        if t is None:
            t = time()

        # filter invalid data
        if self.invalid_value is not None:
            invalid_indices = np.where(x == self.invalid_value)
            x[invalid_indices] = self.x_prev[invalid_indices]

        t_e = max(t - self.t_prev, EPS)
        t_e = np.full(x.shape, t_e)

        # The filtered derivative of the signal.
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
