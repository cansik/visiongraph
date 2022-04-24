from argparse import ArgumentParser, Namespace
from typing import Optional, Any

import cv2
import numpy as np
import vector
from vector import Vector2D, Vector3D

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.input.BaseDepthCamera import BaseDepthCamera
from visiongraph.result.ArUcoCameraPose import ArUcoCameraPose
from visiongraph.result.ArUcoMarkerDetection import ArUcoMarkerDetection


class ArUcoCameraPoseEstimator(VisionEstimator[Optional[ArUcoCameraPose]]):
    def __init__(self, camera: BaseDepthCamera,
                 aruco_config: int = cv2.aruco.DICT_6X6_50,
                 marker_size_in_m: float = 1.0,
                 marker_height_in_m: float = 1.5):
        self.camera = camera
        self.aruco_config: int = aruco_config

        self.marker_size_in_m: float = marker_size_in_m
        self.marker_height_in_m: float = marker_height_in_m

        self.aruco_dict: Optional[Any] = None
        self.aruco_params: Optional[Any] = None

    def setup(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_config)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

    def process(self, data: np.ndarray) -> Optional[ArUcoCameraPose]:
        # find ArUco markers
        (corners, ids, rejected) = cv2.aruco.detectMarkers(data, self.aruco_dict, parameters=self.aruco_params)

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

        return ArUcoCameraPose(Vector3D(), Vector3D(), marker)

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
