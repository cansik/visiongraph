from typing import Tuple


class BoundingBox:
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

    def center(self) -> Tuple[float, float]:
        return self.x_min + self.width * 0.5, self.y_min + self.height * 0.5

