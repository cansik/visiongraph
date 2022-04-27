import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import cv2
import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.CameraPoseResult import CameraPoseResult


class CameraChessboardCalibrator(VisionEstimator[Optional[CameraPoseResult]]):
    def __init__(self, max_samples: int = -1):
        self.max_samples = max_samples

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((6 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

        self.image_size: Optional[Tuple[int, int]] = None

        self.pose_result: Optional[CameraPoseResult] = None

    def setup(self):
        pass

    def process(self, data: np.ndarray) -> Optional[CameraPoseResult]:
        if self.pose_result is not None:
            return self.pose_result

        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret:
            self.objpoints.append(self.objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.imgpoints.append(corners)

            self.image_size = gray.shape[::-1]

            # annotate
            cv2.drawChessboardCorners(data, (7, 6), corners2, ret)

        if 0 < self.max_samples <= self.sample_count:
            return self.calibrate()

        return None

    def calibrate(self) -> Optional[CameraPoseResult]:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                           self.image_size, None, None)

        if ret:
            logging.info("Camera calibrated")
            intrinsics = CameraIntrinsics(mtx, dist)
            self.pose_result = CameraPoseResult(intrinsics)
            return self.pose_result

        logging.warning(f"Could not calibrate camera with {self.sample_count} samples.")
        return None

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @property
    def sample_count(self):
        return len(self.imgpoints)
