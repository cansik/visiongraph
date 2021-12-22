import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import cv2
import numpy as np

from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.util.CollectionUtils import default_value_dict
from visiongraph.util.MathUtils import transform_coordinates, constrain
from visiongraph.util.TimeUtils import current_millis

import pyk4a
from pyk4a import PyK4A, PyK4ACapture, Config


class AzureKinectInput(BaseDepthInput):
    HeightToResolutionMapping = default_value_dict(pyk4a.ColorResolution.RES_720P,
                                                   {
                                                       720: pyk4a.ColorResolution.RES_720P,
                                                       1080: pyk4a.ColorResolution.RES_1080P,
                                                       1440: pyk4a.ColorResolution.RES_1440P,
                                                       1536: pyk4a.ColorResolution.RES_1536P,
                                                       2160: pyk4a.ColorResolution.RES_2160P,
                                                       3072: pyk4a.ColorResolution.RES_3072P,
                                                   })

    def __init__(self):
        super().__init__()
        self.use_infrared = False
        self.sync_frames = True

        self.min_clipping: Optional[int] = None
        self.max_clipping: Optional[int] = None
        self.color_map: Optional[int] = cv2.COLORMAP_JET

        self.device: Optional[PyK4A] = None
        self.capture: Optional[PyK4ACapture] = None

    def setup(self):
        config = Config()
        config.color_resolution = AzureKinectInput.HeightToResolutionMapping[self.height]
        config.depth_mode = pyk4a.DepthMode.OFF
        config.synchronized_images_only = False

        if self.use_infrared:
            config.depth_mode = config.depth_mode.PASSIVE_IR
            config.synchronized_images_only = self.sync_frames

        if self.enable_depth:
            config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
            config.synchronized_images_only = self.sync_frames

        self.device = PyK4A(config)
        self.device.start()

    def read(self) -> (int, Optional[np.ndarray]):
        self.capture = self.device.get_capture()
        time_stamp = current_millis()

        if self.enable_depth and self.use_depth_as_input:
            depth = self.capture.depth
            image = self._colorize(depth, (self.min_clipping, self.max_clipping), self.color_map)
        else:
            if self.use_infrared:
                image = self._colorize(self.capture.ir, (None, None), None)
            else:
                image = self.capture.color
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if image is None:
            logging.warning("could not read frame.")
            return self._post_process(time_stamp, None)

        return self._post_process(time_stamp, image)

    def release(self):
        self.device.stop()

    def distance(self, x: float, y: float) -> float:
        depth_frame = self.capture.depth
        h, w = depth_frame.shape[:2]

        x, y = transform_coordinates(x, y, self.rotate, self.flip)

        ix = round(constrain(w * x, upper=w - 1))
        iy = round(constrain(h * y, upper=h - 1))

        # convert mm into m
        return depth_frame[iy, ix] / 1000

    @staticmethod
    def _colorize(image: np.ndarray,
                  clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
                  colormap: Optional[int] = None) -> np.ndarray:
        if clipping_range[0] or clipping_range[1]:
            img = image.clip(clipping_range[0], clipping_range[1])
        else:
            img = image.copy()
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if colormap is not None:
            img = cv2.applyColorMap(img, colormap)
        return img

    @property
    def depth_map(self) -> np.ndarray:
        return self._colorize(self.capture.depth, (self.min_clipping, self.max_clipping), self.color_map)

    def configure(self, args: Namespace):
        super().configure(args)
        self.use_infrared = args.infrared

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
