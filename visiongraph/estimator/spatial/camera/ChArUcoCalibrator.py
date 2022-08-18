import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import cv2
import numpy as np
from cv2 import aruco

from visiongraph.estimator.spatial.camera.BoardCameraCalibrator import BoardCameraCalibrator
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.CameraPoseResult import CameraPoseResult


class ChArUcoCalibrator(BoardCameraCalibrator):
    def __init__(self, rows: int, columns: int,
                 marker_length_in_m: float,
                 square_length_in_m: float,
                 aruco_config: int = aruco.DICT_4X4_50,
                 max_samples: int = -1, ):
        super().__init__(rows, columns, max_samples)

        self.marker_length_in_m = marker_length_in_m
        self.square_length_in_m = square_length_in_m
        self.aruco_config = aruco_config

        self.board: Optional[aruco.CharucoBoard] = None
        self.aruco_dict: Optional[int] = None
        self.aruco_params: Optional[aruco.DetectorParameters] = None

        self.corners = []
        self.ids = []

        self.image_size: Optional[Tuple[int, int]] = None

        self.pose_result: Optional[CameraPoseResult] = None

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        self._sample_count = 0

    def setup(self):
        self.corners = []
        self.ids = []
        self._sample_count = 0

        self.aruco_dict = aruco.Dictionary_get(self.aruco_config)
        self.aruco_params = aruco.DetectorParameters_create()
        self.board = aruco.CharucoBoard_create(self.rows, self.columns,
                                               self.square_length_in_m, self.marker_length_in_m,
                                               self.aruco_dict)

    def process(self, data: np.ndarray) -> Optional[CameraPoseResult]:
        if self.pose_result is not None:
            return self.pose_result

        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # find markers
        corners, ids, rejected_points = aruco.detectMarkers(gray, self.aruco_dict)

        if len(corners) > 0:
            self.image_size = gray.shape[::-1]

            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=self.criteria)
            res2 = aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                self.corners.append(res2[1])
                self.ids.append(res2[2])

            self._sample_count += 1

        if 0 < self.max_samples <= self.sample_count:
            return self.calibrate()

        return None

    def calibrate(self) -> Optional[CameraPoseResult]:
        camera_mat_init = np.array([[1000., 0., self.image_size[0] / 2.],
                                    [0., 1000., self.image_size[1] / 2.],
                                    [0., 0., 1.]])

        dist_coeffs_init = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        (ret, camera_matrix, distortion_coefficients,
         rotation_vectors, translation_vectors,
         std_deviations_intrinsics, std_deviations_extrinsics,
         per_view_errors) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=self.corners,
            charucoIds=self.ids,
            board=self.board,
            imageSize=self.image_size,
            cameraMatrix=camera_mat_init,
            distCoeffs=dist_coeffs_init,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        if ret:
            intrinsics = CameraIntrinsics(camera_matrix, distortion_coefficients)
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
        return self._sample_count
