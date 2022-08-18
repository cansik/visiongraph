from argparse import ArgumentParser, Namespace
from typing import Optional

import cv2
import numpy as np
import vector
from cv2 import aruco

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.ArUcoCameraPose import ArUcoCameraPose
from visiongraph.result.ArUcoMarkerDetection import ArUcoMarkerDetection


class ArUcoCameraPoseEstimator(VisionEstimator[Optional[ArUcoCameraPose]]):
    def __init__(self,
                 camera_matrix: np.ndarray,
                 fisheye_distortion: np.ndarray,
                 aruco_config: int = aruco.DICT_6X6_50,
                 marker_length_in_m: float = 0.1):
        self.camera_matrix = camera_matrix
        self.fisheye_distortion = fisheye_distortion

        self.aruco_config: int = aruco_config

        self.marker_size_in_m: float = marker_length_in_m

        self.aruco_dict: Optional[int] = None
        self.aruco_params: Optional[aruco.DetectorParameters] = None

    def setup(self):
        self.aruco_dict = aruco.Dictionary_get(self.aruco_config)
        self.aruco_params = aruco.DetectorParameters_create()

    def process(self, data: np.ndarray) -> Optional[ArUcoCameraPose]:
        # find ArUco markers
        (corners, ids, rejected) = aruco.detectMarkers(data, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) == 0:
            return None

        # select first marker
        ids = ids.flatten()
        marker_corner, marker_id = list(zip(corners, ids))[0]

        # top-left, top-right, bottom-right, and bottom-left order
        corners = marker_corner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        marker = ArUcoMarkerDetection(marker_id,
                                      vector.obj(x=topLeft[0], y=topLeft[1]),
                                      vector.obj(x=topRight[0], y=topRight[1]),
                                      vector.obj(x=bottomRight[0], y=bottomRight[1]),
                                      vector.obj(x=bottomLeft[0], y=bottomLeft[1]))

        # estimate pose
        rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers([marker_corner],
                                                                                 self.marker_size_in_m,
                                                                                 self.camera_matrix,
                                                                                 self.fisheye_distortion)

        cv2.drawFrameAxes(data, self.camera_matrix, self.fisheye_distortion, rotation_vector, translation_vector, 0.1)

        #
        return ArUcoCameraPose(position=vector.obj(x=translation_vector[0, 0, 0],
                                                   y=translation_vector[0, 0, 1],
                                                   z=translation_vector[0, 0, 2]),
                               rotation=vector.obj(x=rotation_vector[0, 0, 0],
                                                   y=rotation_vector[0, 0, 1],
                                                   z=rotation_vector[0, 0, 2]),
                               marker=marker)

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
