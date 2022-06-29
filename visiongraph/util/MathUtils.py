import math
from typing import Optional, Tuple

import cv2


def transform_coordinates(x: float, y: float, rotate: Optional[int], flip: Optional[int]) -> Tuple[float, float]:
    nx, ny = x, y

    if rotate == cv2.ROTATE_90_CLOCKWISE or (rotate == cv2.ROTATE_90_COUNTERCLOCKWISE and flip == 1):
        nx = y
        ny = 1.0 - x
    elif rotate == cv2.ROTATE_90_COUNTERCLOCKWISE or (rotate == cv2.ROTATE_90_CLOCKWISE and flip == 1):
        nx = 1.0 - y
        ny = x
    elif rotate == cv2.ROTATE_180:
        nx = 1.0 - x
        ny = 1.0 - y

    if flip == 1:
        nx = 1.0 - nx
    elif flip == 0:
        ny = 1.0 - ny

    return nx, ny


def constrain(value: float, lower: float = 0, upper: float = 1) -> float:
    return max(min(value, upper), lower)


def map_value(value, istart, istop, ostart, ostop) -> float:
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def rotate_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

    def average(self):
        return float(self.sum) / max(len(self.values), 1)
