from typing import Sequence, Union

import numpy as np
import vector

from visiongraph.model.geometry.Size2D import Size2D


class BoundingBox2D:
    def __init__(self, x_min: float, y_min: float, width: float, height: float):
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def __iter__(self):
        yield self.x_min
        yield self.y_min
        yield self.width
        yield self.height

    @property
    def center(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min + self.width * 0.5, y=self.y_min + self.height * 0.5)

    @property
    def top_left(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min, y=self.y_min)

    @property
    def top_right(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min + self.width, y=self.y_min)

    @property
    def bottom_right(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min + self.width, y=self.y_min + self.height)

    @property
    def bottom_left(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min, y=self.y_min + self.height)

    @property
    def x_max(self) -> float:
        return self.x_min + self.width

    @property
    def y_max(self):
        return self.y_min + self.height

    @property
    def size(self) -> Size2D:
        return Size2D(self.width, self.height)

    def to_array(self, tl_br_format: bool = False) -> np.ndarray:
        if tl_br_format:
            return np.array([self.x_min, self.y_min, self.x_min + self.width, self.y_min + self.height])
        return np.array([self.x_min, self.y_min, self.width, self.height])

    def scale(self, width: float, height: float) -> "BoundingBox2D":
        return BoundingBox2D(
            self.x_min * width,
            self.y_min * height,
            self.width * width,
            self.height * height)

    def scale_with(self, size: Size2D) -> "BoundingBox2D":
        return self.scale(size.width, size.height)

    def scale_centered(self, width: float, height: float) -> "BoundingBox2D":
        dx = self.width * width
        dy = self.height * height
        return self.add_border(dx, dy)

    def add_border(self, dx: float, dy: float) -> "BoundingBox2D":
        return BoundingBox2D(
            self.x_min - (dx * 0.5),
            self.y_min - (dy * 0.5),
            self.width + dx,
            self.height + dy)

    def add_border_with(self, size: Size2D) -> "BoundingBox2D":
        return self.add_border(size.width, size.height)

    @staticmethod
    def from_array(data: Union[Sequence, np.ndarray], tl_br_format: bool = False):
        if isinstance(data, np.ndarray):
            data = data.flat

        if tl_br_format:
            return BoundingBox2D(data[0], data[1], data[2] - data[0], data[3] - data[1])

        return BoundingBox2D(data[0], data[1], data[2], data[3])

    @staticmethod
    def from_image(image: np.ndarray):
        h, w = image.shape[:2]
        return BoundingBox2D(0, 0, float(w), float(h))

    @staticmethod
    def from_kernel(x: int, y: int, kernel_size: int):
        shift = kernel_size // 2
        return BoundingBox2D(x - shift, y - shift, kernel_size, kernel_size)

    def intersection_over_union(self, box: "BoundingBox2D") -> float:
        box_a = self.to_array(tl_br_format=True)
        box_b = box.to_array(tl_br_format=True)

        # determine the (x, y)-coordinates of the intersection rectangle
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou

    def contains(self, p: vector.Vector2D) -> bool:
        if self.x_min < p.x < self.x_max:
            if self.y_min < p.y < self.y_max:
                return True
        return False

    def __repr__(self):
        return f"BoundingBox2D(x={self.x_min:.4f}, y={self.y_min:.4f}, w={self.width:.4f}, h={self.height:.4f})"
