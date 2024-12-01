import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import cv2
import numpy as np

from visiongraph.estimator.spatial.camera.BoardCameraCalibrator import BoardCameraCalibrator
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.CameraPoseResult import CameraPoseResult


class ChessboardCalibrator(BoardCameraCalibrator):
    """
    A class to calibrate a camera using a chessboard pattern.
    """

    def __init__(self, columns: int, rows: int, max_samples: int = -1):
        """
        Initializes the ChessboardCalibrator object with the number of rows and columns,
        as well as the maximum number of samples.

        :param columns: The number of columns in the chessboard pattern.
        :param rows: The number of rows in the chessboard pattern.
        :param max_samples: The maximum number of samples. Defaults to -1.
        """
        super().__init__(rows, columns, max_samples)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((self.rows * self.columns, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.rows, 0:self.columns].T.reshape(-1, 2)

        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.

        self.image_size: Optional[Tuple[int, int]] = None

        self.pose_result: Optional[CameraPoseResult] = None

    def setup(self):
        """
        Sets up the ChessboardCalibrator object.
        """
        pass

    def process(self, data: np.ndarray) -> Optional[CameraPoseResult]:
        """
        Processes a frame of image data to detect chessboard corners.

        :param data: The input image frame.

        :return: The calibrated camera pose if detected, otherwise None.
        """
        self.board_detected = False

        if self.pose_result is not None:
            return self.pose_result

        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.rows, self.columns), None)

        if ret:
            self.obj_points.append(self.objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.img_points.append(corners)

            self.image_size = gray.shape[::-1]

            # annotate
            cv2.drawChessboardCorners(data, (self.rows, self.columns), corners2, ret)

            self.board_detected = True

        if 0 < self.max_samples <= self.sample_count:
            return self.calibrate()

        return None

    def calibrate(self) -> Optional[CameraPoseResult]:
        """
        Calculates the camera pose using the detected chessboard corners.

        :return: The calibrated camera pose if successful, otherwise None.
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                           self.image_size, None, None)

        if ret:
            logging.info("Camera calibrated")
            intrinsics = CameraIntrinsics(mtx, dist)
            self.pose_result = CameraPoseResult(intrinsics)

            mean_error = 0
            for i in range(len(self.obj_points)):
                imgpoints2, _ = cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            print(f"Total error: {mean_error / len(self.obj_points)}")

            return self.pose_result

        logging.warning(f"Could not calibrate camera with {self.sample_count} samples.")
        return None

    def release(self):
        """
        Releases any system resources used by the ChessboardCalibrator object.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the ChessboardCalibrator object based on user input.

        :param args: The parsed command line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argument parser for the ChessboardCalibrator class.

        :param parser: The argument parser object.
        """
        pass

    @property
    def sample_count(self):
        """
        Gets the number of samples used in the camera calibration process.

        :return: The number of samples.
        """
        return len(self.img_points)
