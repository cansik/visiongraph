from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Tuple, Optional

import cv2
import numpy as np

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.model.CameraStreamType import CameraStreamType
from visiongraph.util import MathUtils


class BaseDepthCamera(BaseCamera, BaseDepthInput, ABC):
    """
    Abstract base class for depth cameras that handle both color and depth input streams.
    """

    def __init__(self):
        """
        Initializes the BaseDepthCamera with default settings.
        """
        super().__init__()
        self.use_infrared = False

    def configure(self, args: Namespace):
        """
        Configures the camera settings based on command line arguments.

        :param args: The command line arguments namespace.
        """
        super().configure(args)
        self.use_infrared = args.infrared

    def _calculate_depth_coordinates(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        """
        Calculates depth coordinates from given normalized coordinates.

        :param x: The x coordinate (normalized).
        :param y: The y coordinate (normalized).
        :param width: The width of the image.
        :param height: The height of the image.

        :return: The calculated pixel coordinates (ix, iy).
        """
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
        """
        Colorizes a depth image within a specified clipping range.

        :param image: The image to colorize.
        :param clipping_range: The clipping range for normalization.
        :param colormap: The OpenCV colormap to apply.

        :return: The colorized image.
        """
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

    @abstractmethod
    def pre_process_image(self, image: np.ndarray,
                          stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Pre-processes the input image based on the stream type.

        :param image: The image to be pre-processed.
        :param stream_type: The type of the camera stream (default is CameraStreamType.Color).

        :return: The pre-processed image or None if unprocessable.
        """
        return image

    @abstractmethod
    def get_raw_image(self, stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Retrieves the raw image from the camera based on the stream type.

        :param stream_type: The type of the camera stream (default is CameraStreamType.Color).

        :return: The raw image or None if unavailable.
        """
        pass

    def get_image(self, stream_type: CameraStreamType = CameraStreamType.Color,
                  pre_processed: bool = True, post_processed: bool = True) -> Optional[np.ndarray]:
        """
        Retrieves and processes the image from the camera stream.

        :param stream_type: The type of the camera stream (default is CameraStreamType.Color).
        :param pre_processed: Whether to pre-process the image (default is True).
        :param post_processed: Whether to post-process the image (default is True).

        :return: The processed image or None if unavailable.
        """
        frame = self.get_raw_image(stream_type)

        if frame is None:
            return None

        # apply camera pre-processing
        if pre_processed:
            frame = self.pre_process_image(frame, stream_type)

        # apply base camera post-processing
        if post_processed:
            _, frame = self._post_process(0, frame)
            return frame

        return frame

    @property
    def color_image(self) -> Optional[np.ndarray]:
        """
        Returns the processed color image.

        :return: The processed color image or None if unavailable.
        """
        return self.get_image(CameraStreamType.Color, True, True)

    @property
    def depth_image(self) -> Optional[np.ndarray]:
        """
        Returns the processed depth image.

        :return: The processed depth image or None if unavailable.
        """
        return self.get_image(CameraStreamType.Depth, True, True)

    @property
    def infrared_image(self) -> Optional[np.ndarray]:
        """
        Returns the processed infrared image.

        :return: The processed infrared image or None if unavailable.
        """
        return self.get_image(CameraStreamType.Infrared, True, True)

    @property
    def raw_color_image(self) -> Optional[np.ndarray]:
        """
        Returns the raw color image from the camera.

        :return: The raw color image or None if unavailable.
        """
        return self.get_raw_image(CameraStreamType.Color)

    @property
    def raw_depth_image(self) -> Optional[np.ndarray]:
        """
        Returns the raw depth image from the camera.

        :return: The raw depth image or None if unavailable.
        """
        return self.get_raw_image(CameraStreamType.Depth)

    @property
    def raw_infrared_image(self) -> Optional[np.ndarray]:
        """
        Returns the raw infrared image from the camera.

        :return: The raw infrared image or None if unavailable.
        """
        return self.get_raw_image(CameraStreamType.Infrared)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds camera-specific parameters to the argument parser.

        :param parser: The argument parser instance.
        """
        super(BaseDepthCamera, BaseDepthCamera).add_params(parser)
        BaseDepthInput.add_params(parser)

        try:
            parser.add_argument("-ir", "--infrared", action="store_true",
                                help="Use infrared as input stream.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex

    @property
    def is_playback(self) -> bool:
        """
        Indicates whether the camera is in playback mode.

        :return: False since this is a real-time camera.
        """
        return False
