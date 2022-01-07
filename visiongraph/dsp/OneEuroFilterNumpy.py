"""
Source: https://github.com/HoBeom/OneEuroFilter-Numpy
MIT License
"""
from typing import Optional

import numpy as np
from time import time


def _smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilterNumpy:
    def __init__(self, x0: np.ndarray, t0: Optional[float] = None, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.data_shape = x0.shape
        self.min_cutoff = np.full(x0.shape, min_cutoff)
        self.beta = np.full(x0.shape, beta)
        self.d_cutoff = np.full(x0.shape, d_cutoff)

        # Previous values.
        self.x_prev = x0.astype(np.float)
        self.dx_prev = np.full(x0.shape, dx0)
        self.t_prev = time() if t0 is None else t0

    def __call__(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Compute the filtered signal."""
        assert x.shape == self.data_shape

        if t is None:
            t = time()

        t_e = t - self.t_prev
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
