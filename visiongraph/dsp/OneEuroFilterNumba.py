"""
Source: https://github.com/HoBeom/OneEuroFilter-Numpy
Adapted to work with numba
MIT License
"""

from time import time
from typing import Optional

import numpy as np
from numba import njit

from visiongraph import OneEuroFilterNumpy


@njit()
def _smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


@njit()
def _exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


@njit()
def _apply_filter(x, t, x_prev, t_prev, dx_prev, min_cutoff, beta, d_cutoff):
    """Compute the filtered signal."""

    t_e = t - t_prev
    t_e = np.full(x.shape, t_e)

    # The filtered derivative of the signal.
    a_d = _smoothing_factor(t_e, d_cutoff)
    dx = (x - x_prev) / t_e
    dx_hat = _exponential_smoothing(a_d, dx, dx_prev)

    # The filtered signal.
    cutoff = min_cutoff + beta * np.abs(dx_hat)
    a = _smoothing_factor(t_e, cutoff)
    x_hat = _exponential_smoothing(a, x, x_prev)

    # Memorize the previous values.
    x_prev = x_hat
    dx_prev = dx_hat
    t_prev = t

    return x_hat, x_prev, dx_prev, t_prev


class OneEuroFilterNumba(OneEuroFilterNumpy):
    def __call__(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Compute the filtered signal."""
        assert x.shape == self.data_shape

        if t is None:
            t = time()

        x_hat, x_prev, dx_prev, t_prev = _apply_filter(x, t, self.x_prev, self.t_prev, self.dx_prev,
                                                       self.min_cutoff, self.beta, self.d_cutoff)

        self.x_prev = x_prev
        self.dx_prev = dx_prev
        self.t_prev = t_prev

        return x_hat
