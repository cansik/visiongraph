import math
from typing import Optional, Tuple, Sequence

import cv2
import numpy as np


def transform_coordinates(x: float, y: float, rotate: Optional[int], flip: Optional[int]) -> Tuple[float, float]:
    """
    Transforms the (x, y) coordinates based on specified rotation and flipping.

    :param x: The x-coordinate to transform.
    :param y: The y-coordinate to transform.
    :param rotate: The rotation value that determines how to rotate the coordinates.
    :param flip: The flip value that determines how to flip the coordinates.

    :return: The transformed (x, y) coordinates.
    """
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
    """
    Constrains a value to be within the specified lower and upper bounds.

    :param value: The value to constrain.
    :param lower: The lower bound.
    :param upper: The upper bound.

    :return: The constrained value.
    """
    return max(min(value, upper), lower)


def map_value(value, istart, istop, ostart, ostop) -> float:
    """
    Maps a value from one range to another.

    :param value: The input value to map.
    :param istart: The start of the input range.
    :param istop: The end of the input range.
    :param ostart: The start of the output range.
    :param ostop: The end of the output range.

    :return: The mapped value in the new range.
    """
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def rotate_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    :param origin: The (x, y) coordinates of the origin point.
    :param point: The (x, y) coordinates of the point to rotate.
    :param angle: The angle in radians by which to rotate the point.

    :return: The (x, y) coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class StreamingMovingAverage:
    def __init__(self, window_size):
        """
        Initializes the StreamingMovingAverage object with a specific window size.

        :param window_size: The size of the moving average window.
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        """
        Processes a new value and updates the moving average.

        :param value: The new value to process.

        :return: The updated moving average.
        """
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

    def average(self):
        """
        Calculates the current moving average.

        :return: The moving average value.
        """
        return float(self.sum) / max(len(self.values), 1)


def intersection_over_union(a: Sequence[float], b: Sequence[float], epsilon: float = 1e-5) -> float:
    """
    Given two boxes `a` and `b` defined as a list of four numbers:
        [x1,y1,x2,y2]
    where:
        x1,y1 represent the upper left corner
        x2,y2 represent the lower right corner
    It returns the Intersection over Union (IoU) score for these two boxes.

    Source: http://ronny.rest/tutorials/module/localization_001/iou/

    :param a: The first box defined by [x1,y1,x2,y2].
    :param b: The second box defined by [x1,y1,x2,y2].
    :param epsilon: Small value to prevent division by zero. Defaults to 1e-5.

    :return: The Intersection over Union score.
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
    """
    Applies the sigmoid function to a given value.

    :param x: The input value to apply the sigmoid function to.

    """
    return 1 / (1 + np.exp(-x))


def decompose_transformation_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes a 4x4 transformation matrix into rotation, translation, and scale components.

    :param matrix: A 4x4 transformation matrix.
    :returns: A triple containing the 3x3 rotation matrix, 3x1 translation vector, and 3x1 scale vector.
    """
    # Ensure the input is a 4x4 matrix
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    # Extract the rotation matrix (upper-left 3x3 submatrix)
    rotation_matrix = matrix[:3, :3]

    # Extract the translation vector (upper-right 3x1 vector)
    translation_vector = matrix[:3, 3]

    # Compute the scale factors from the norms of the rotation matrix rows or columns
    scale_factors = np.linalg.norm(rotation_matrix, axis=0)  # Column-wise norms

    # Normalize the rotation matrix to remove scale
    rotation_matrix_normalized = rotation_matrix / scale_factors

    return rotation_matrix_normalized, translation_vector, scale_factors
