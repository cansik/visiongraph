from typing import Optional

import cv2
import numpy as np

from visiongraph.model.DepthBuffer import DepthBuffer
from visiongraph.result.ImageResult import ImageResult
from visiongraph.util.MathUtils import constrain


class DepthMap(DepthBuffer, ImageResult):
    def __init__(self, buffer: np.ndarray):
        self._buffer = buffer
        super().__init__(self.apply_colormap())

    @property
    def depth_buffer(self) -> np.ndarray:
        return self._buffer

    @property
    def depth_map(self) -> np.ndarray:
        return self.output

    def apply_colormap(self, color_map=cv2.COLORMAP_INFERNO) -> np.ndarray:
        norm_buffer = self.normalize_buffer()
        self.output = cv2.applyColorMap(norm_buffer, colormap=color_map)
        return self.output

    def normalize_buffer(self, bit_depth: int = 8,
                         depth_min: Optional[float] = None,
                         depth_max: Optional[float] = None) -> np.ndarray:
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
        h, w = self._buffer.shape[:2]

        ix = constrain(round(w * x, 0), w - 1)
        iy = constrain(round(h * y, 0), h - 1)

        return float(self._buffer[iy, ix, 0])
