from typing import Tuple

import cv2
import numpy as np

COLOR_SEQUENCE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (255, 255, 255)
]

AXIS_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0)
]


def draw_axis(image: np.ndarray, rotation: Tuple[float, float, float],
              center: Tuple[float, float], length: float = 0.1):
    h, w = image.shape[:2]
    rays = np.asarray([[length, 0, 0], [0, length, 0], [0, 0, -length]])

    for i, p in enumerate(rays):
        color = AXIS_COLORS[i]
        pp = apply_axis_rotation(p, *rotation)

        x = (pp[0] + center[0]) * w
        y = (-pp[1] + center[1]) * h

        cv2.line(image, (round(center[0] * w), round(center[1] * h)),
                 (round(x), round(y)), color=color, thickness=2)


def apply_axis_rotation(v: Tuple[float, float, float], rx: float, ry: float, rz: float) -> Tuple[float, float, float]:
    """
    Apply rotation to a vector
    :param v: Input vector x, y, z
    :param rx: Rotation x in degree
    :param ry: Rotation y in degree
    :param rz: Rotation z in degree
    :return: Rotated vector coordinates
    """

    v = np.asarray(v)

    # convert to radians
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    mx = np.asarray([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])

    my = np.asarray([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    mz = np.asarray([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])

    result = v.dot(mx).dot(my).dot(mz)
    return result
