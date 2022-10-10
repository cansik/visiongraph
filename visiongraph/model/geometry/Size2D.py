from typing import Union, Sequence

import numpy as np


class Size2D:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def __iter__(self):
        yield self.width
        yield self.height

    def scale(self, width: float, height: float) -> "Size2D":
        return Size2D(
            self.width * width,
            self.height * height)

    @staticmethod
    def from_array(data: Union[Sequence, np.ndarray]):
        if isinstance(data, np.ndarray):
            data = data.flat

        return Size2D(data[0], data[1])

    @staticmethod
    def from_image(image: np.ndarray):
        h, w = image.shape[:2]
        return Size2D(float(w), float(h))

    def __repr__(self):
        return f"Size2D(w={self.width:.4f}, h={self.height:.4f})"
