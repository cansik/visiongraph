from abc import ABC
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Tuple, Optional

import cv2
import numpy as np

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.util import MathUtils


class BaseDepthCamera(BaseCamera, BaseDepthInput, ABC):
    def __init__(self):
        super().__init__()

        self.use_infrared = False

    def configure(self, args: Namespace):
        super().configure(args)

        self.use_infrared = args.infrared

    def _calculate_depth_coordinates(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        x, y = MathUtils.transform_coordinates(x, y, self.rotate, self.flip)

        if self.crop is not None:
            norm_crop = self.crop.scale(1.0 / width, 1.0 / height)
            x = MathUtils.map_value(x, 0.0, 1.0, norm_crop.x_min, norm_crop.x_max)
            y = MathUtils.map_value(y, 0.0, 1.0, norm_crop.y_min, norm_crop.y_max)

        ix, iy = width * x, height * y

        ix = round(MathUtils.constrain(ix, upper=width - 1))
        iy = round(MathUtils.constrain(iy, upper=height - 1))

        return ix, iy

    @staticmethod
    def _colorize(image: np.ndarray,
                  clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
                  colormap: Optional[int] = None) -> np.ndarray:
        if clipping_range[0] is not None and clipping_range[1] is not None:
            low, high = clipping_range
            delta = high - low

            img = image.clip(low, high)
            img = (((img - low) / delta) * 255).astype(np.uint8)
        else:
            img = image
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if colormap is not None:
            img = cv2.applyColorMap(img, colormap)
        return img

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(BaseDepthCamera, BaseDepthCamera).add_params(parser)
        BaseDepthInput.add_params(parser)

        try:
            parser.add_argument("-ir", "--infrared", action="store_true",
                                help="Use infrared as input stream.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex
