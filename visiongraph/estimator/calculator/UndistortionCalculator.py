from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, Any

import cv2
import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.model.CameraIntrinsics import CameraIntrinsics


class UndistortionCalculator(VisionEstimator[np.ndarray]):
    """
    Calculates the optimal camera matrix for undistortion and applies rectification.
    """

    def __init__(self, intrinsics: CameraIntrinsics, width: int = 0, height: int = 0):
        """
        Initializes the UndistortionCalculator with camera intrinsics and image dimensions.

        :param intrinsics: The intrinsic camera parameters.
        :param width: The image width. Defaults to 0.
        :param height: The image height. Defaults to 0.
        """
        self.intrinsics = intrinsics

        self.width = width
        self.height = height

        self.new_camera_matrix: Optional[np.ndarray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None

        self.rectify_map_x: Optional[Any] = None
        self.rectify_map_y: Optional[Any] = None

    def setup(self):
        """
        Sets up the calculator by calculating the optimal camera matrix if the image dimensions are known.
        """
        if self.width > 0 and self.height > 0:
            self.calculate_optimal_camera_matrix()

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Applies undistortion to the input data.

        :param data: The input image data.

        :return: The undistorted image data.
        """
        h, w = data.shape[:2]

        if h != self.height or w != self.width:
            self.width = w
            self.height = h
            self.calculate_optimal_camera_matrix()

        dst = cv2.remap(data, self.rectify_map_x, self.rectify_map_y, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def release(self):
        """
        Releases any resources used by the calculator.
        """
        pass

    def calculate_optimal_camera_matrix(self) -> None:
        """
        Calculates the optimal camera matrix using OpenCV's getOptimalNewCameraMatrix function.

        """
        w = self.width
        h = self.height
        mat, roi = cv2.getOptimalNewCameraMatrix(self.intrinsics.intrinsic_matrix,
                                                 self.intrinsics.distortion_coefficients,
                                                 (w, h), 1, (w, h))
        self.new_camera_matrix = mat
        self.roi = roi

        self.rectify_map_x, self.rectify_map_y = cv2.initUndistortRectifyMap(self.intrinsics.intrinsic_matrix,
                                                                             self.intrinsics.distortion_coefficients,
                                                                             None, self.new_camera_matrix, (w, h), 5)

    def configure(self, args: Namespace) -> None:
        """
        Configures the calculator based on command line arguments.

        :param args: The parsed command line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser) -> None:
        """
        Adds parameters to the parser for configuring the calculator.

        :param parser: The parser to add parameters to.
        """
        pass
