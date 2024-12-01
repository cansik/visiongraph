from typing import Optional

import cv2
import numpy as np

from visiongraph.model.DepthBuffer import DepthBuffer
from visiongraph.result.ImageResult import ImageResult
from visiongraph.util.MathUtils import constrain


class DepthMap(DepthBuffer, ImageResult):
    """
    A class representing a depth map image result.
    """

    def __init__(self, buffer: np.ndarray):
        """
        Initializes the DepthMap object with the given depth buffer.

        :param buffer: The input depth buffer array.
        """
        self._buffer = buffer
        super().__init__(self.apply_colormap())

    @property
    def depth_buffer(self) -> np.ndarray:
        """
        Gets the underlying depth buffer array.

        :return: The input depth buffer array.
        """
        return self._buffer

    @property
    def depth_map(self) -> np.ndarray:
        """
        Gets the output depth map image result.

        :return: The output depth map image array.
        """
        return self.output

    def apply_colormap(self, color_map=cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Applies a colormap to the depth buffer and updates the output.

        :param color_map: The colormap index. Defaults to cv2.COLORMAP_INFERNO.

        :return: The colored depth map image array.
        """
        norm_buffer = self.normalize_buffer()
        self.output = cv2.applyColorMap(norm_buffer, colormap=color_map)
        return self.output

    def normalize_buffer(self, bit_depth: int = 8,
                         depth_min: Optional[float] = None,
                         depth_max: Optional[float] = None) -> np.ndarray:
        """
        Normalizes the depth buffer values to a specified range.

        :param bit_depth: The number of bits in the output depth data. Defaults to 8.
        :param depth_min: The minimum allowed depth value. If not provided, uses the actual minimum value.
        :param depth_max: The maximum allowed depth value. If not provided, uses the actual maximum value.

        :return: The normalized depth buffer array.
        """
        max_val = pow(2, bit_depth)

        # normalize prediction
        depth_min = self._buffer.min() if depth_min is None else depth_min
        depth_max = self._buffer.max() if depth_max is None else depth_max

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (self._buffer - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        if bit_depth == 8:
            return out.astype(np.uint8)
        else:
            return out.astype(np.uint16)

    def distance(self, x: float, y: float) -> float:
        """
        Calculates the depth value at a given 2D point.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.

        :return: The corresponding depth value.
        """
        h, w = self._buffer.shape[:2]

        ix = constrain(round(w * x, 0), w - 1)
        iy = constrain(round(h * y, 0), h - 1)

        return float(self._buffer[iy, ix, 0])
