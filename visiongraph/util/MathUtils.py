import math
from typing import Optional, Tuple, Sequence

import cv2
import numpy as np


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


def intersection_over_union(a: Sequence[float], b: Sequence[float], epsilon: float = 1e-5) -> float:
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

        Source: http://ronny.rest/tutorials/module/localization_001/iou/

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
