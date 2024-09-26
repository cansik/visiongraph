"""
Source: https://github.com/HoBeom/OneEuroFilter-Numpy
Adapted to work with numba
MIT License
"""

from time import time
from typing import Optional

import numpy as np
from numba import njit

from visiongraph.dsp.OneEuroFilterNumpy import OneEuroFilterNumpy


@njit()
def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """
    Calculates the smoothing factor based on the time elapsed and cutoff frequency.

    Args:
        t_e (float): Time elapsed.
        cutoff (float): Cutoff frequency.

    Returns:
        float: Smoothing factor.
    """
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


@njit()
def _exponential_smoothing(a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    """
    Applies exponential smoothing to the signal.

    Args:
        a (float): Smoothing factor.
        x (np.ndarray): Signal values.
        x_prev (np.ndarray): Previous signal value.

    Returns:
        np.ndarray: Exponentially smoothed signal.
    """
    return a * x + (1 - a) * x_prev


@njit()
def _apply_filter(x: np.ndarray, t: float, x_prev: np.ndarray, t_prev: float, dx_prev: np.ndarray,
                  min_cutoff: float, beta: float, d_cutoff: float) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    Computes the filtered signal using the OneEuro filter algorithm.

    Args:
        x (np.ndarray): Signal values.
        t (float): Time stamp.
        x_prev (np.ndarray): Previous signal value.
        t_prev (float): Previous time stamp.
        dx_prev (np.ndarray): Previous derivative of the signal.
        min_cutoff (float): Minimum cutoff frequency.
        beta (float): Beta parameter for exponential smoothing.
        d_cutoff (float): Cutoff frequency for derivative.

    Returns:
        np.ndarray: Filtered signal values.
        np.ndarray: Previous filtered signal value.
        np.ndarray: Previous derivative of the signal.
        float: Previous time stamp.
    """
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
        """
        Computes the filtered signal using the OneEuro filter algorithm.

        Args:
            x (np.ndarray): Signal values.
            t (float, optional): Time stamp. Defaults to current time if not provided.

        Returns:
            np.ndarray: Filtered signal values.
        """
        assert x.shape == self.data_shape

        if t is None:
            t = time()

        x_hat, x_prev, dx_prev, t_prev = _apply_filter(x, t, self.x_prev, self.t_prev, self.dx_prev,
                                                       self.min_cutoff, self.beta, self.d_cutoff)

        self.x_prev = x_prev
        self.dx_prev = dx_prev
        self.t_prev = t_prev

        return x_hat
