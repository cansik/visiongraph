from typing import Tuple

import vector


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
    def bottom_right(self) -> vector.Vector2D:
        return vector.obj(x=self.x_min + self.width, y=self.y_min + self.height)

    @property
    def size(self) -> vector.Vector2D:
        return vector.obj(x=self.width, y=self.height)

