from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, Any

import cv2
import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.model.CameraIntrinsics import CameraIntrinsics


class UndistortionCalculator(VisionEstimator[np.ndarray]):
    def __init__(self, intrinsics: CameraIntrinsics, width: int = 0, height: int = 0):
        self.intrinsics = intrinsics

        self.width = width
        self.height = height

        self.new_camera_matrix: Optional[np.ndarray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None

        self.rectify_map_x: Optional[Any] = None
        self.rectify_map_y: Optional[Any] = None

    def setup(self):
        if self.width > 0 and self.height > 0:
            self.calculate_optimal_camera_matrix()

    def process(self, data: np.ndarray) -> np.ndarray:
        h, w = data.shape[:2]

        if h != self.height or w != self.width:
            self.width = w
            self.height = h
            self.calculate_optimal_camera_matrix()

        dst = cv2.remap(data, self.rectify_map_x, self.rectify_map_y, cv2.INTER_LINEAR)

        # dst = cv2.undistort(data,
        #                     self.intrinsics.intrinsic_matrix,
        #                     self.intrinsics.distortion_coefficients,
        #                     None,
        #                     self.new_camera_matrix)

        # crop the image
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def release(self):
        pass

    def calculate_optimal_camera_matrix(self):
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

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
