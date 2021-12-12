from typing import Sequence

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

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


def draw_axis(image: np.ndarray, rotation: vector.Vector3D,
              center: vector.Vector2D, length: float = 0.1):
    h, w = image.shape[:2]
    rays = [vector.obj(x=length, y=0, z=0),
            vector.obj(x=0, y=length, z=0),
            vector.obj(x=0, y=0, z=length)]

    for i, p in enumerate(rays):
        color = AXIS_COLORS[i]
        pp = p.rotate_nautical(np.radians(rotation.z), -np.radians(rotation.y), -np.radians(rotation.x))

        x = (pp.x + center.x) * w
        y = (-pp.y + center.y) * h

        cv2.line(image, (round(center.x * w), round(center.y * h)),
                 (round(x), round(y)), color=color, thickness=2)


def draw_bbox(image: np.ndarray, bbox: BoundingBox2D, color: Sequence[int], thickness: int = 2):
    h, w = image.shape[:2]
    cv2.rectangle(image, (round(bbox.x_min * w), round(bbox.y_min * h)),
                  (round((bbox.x_min + bbox.width) * w), round((bbox.y_min + bbox.height) * h)),
                  color, thickness=thickness)
