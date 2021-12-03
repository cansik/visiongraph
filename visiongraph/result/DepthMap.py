import cv2
import numpy as np

from visiongraph.model.DepthBuffer import DepthBuffer
from visiongraph.result.ImageResult import ImageResult
from visiongraph.util.MathUtils import constrain


class DepthMap(DepthBuffer, ImageResult):
    def apply_colormap(self, color_map=cv2.COLORMAP_INFERNO) -> np.ndarray:
        return cv2.applyColorMap(self.output, colormap=color_map)

    def distance(self, x: float, y: float) -> float:
        h, w = self.output.shape[:2]

        ix = constrain(round(w * x, 0), w - 1)
        iy = constrain(round(h * y, 0), h - 1)

        return float(self.output[iy, ix, 0])
