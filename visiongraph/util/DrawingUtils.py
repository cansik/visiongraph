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

COCO80_COLORS = [
    (0, 113, 188),
    (216, 82, 24),
    (236, 176, 31),
    (125, 46, 141),
    (118, 171, 47),
    (76, 189, 237),
    (161, 19, 46),
    (76, 76, 76),
    (153, 153, 153),
    (255, 0, 0),
    (255, 127, 0),
    (190, 190, 0),
    (0, 255, 0),
    (0, 0, 255),
    (170, 0, 255),
    (84, 84, 0),
    (84, 170, 0),
    (84, 255, 0),
    (170, 84, 0),
    (170, 170, 0),
    (170, 255, 0),
    (255, 84, 0),
    (255, 170, 0),
    (255, 255, 0),
    (0, 84, 127),
    (0, 170, 127),
    (0, 255, 127),
    (84, 0, 127),
    (84, 84, 127),
    (84, 170, 127),
    (84, 255, 127),
    (170, 0, 127),
    (170, 84, 127),
    (170, 170, 127),
    (170, 255, 127),
    (255, 0, 127),
    (255, 84, 127),
    (255, 170, 127),
    (255, 255, 127),
    (0, 84, 255),
    (0, 170, 255),
    (0, 255, 255),
    (84, 0, 255),
    (84, 84, 255),
    (84, 170, 255),
    (84, 255, 255),
    (170, 0, 255),
    (170, 84, 255),
    (170, 170, 255),
    (170, 255, 255),
    (255, 0, 255),
    (255, 84, 255),
    (255, 170, 255),
    (42, 0, 0),
    (84, 0, 0),
    (127, 0, 0),
    (170, 0, 0),
    (212, 0, 0),
    (255, 0, 0),
    (0, 42, 0),
    (0, 84, 0),
    (0, 127, 0),
    (0, 170, 0),
    (0, 212, 0),
    (0, 255, 0),
    (0, 0, 42),
    (0, 0, 84),
    (0, 0, 127),
    (0, 0, 170),
    (0, 0, 212),
    (0, 0, 255),
    (0, 0, 0),
    (36, 36, 36),
    (72, 72, 72),
    (109, 109, 109),
    (145, 145, 145),
    (182, 182, 182),
    (218, 218, 218),
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
