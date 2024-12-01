from argparse import ArgumentParser, Namespace
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from visiongraph.GraphNode import GraphNode
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.face.IrisDistanceResult import IrisDistanceResult, IrisParameter
from visiongraph.util.VectorUtils import landmarks_center_by_indices


class IrisDistanceCalculator(GraphNode[ResultList[BlazeFaceMesh], ResultList[IrisDistanceResult]]):
    """
    A class used to calculate the distance between iris in face detection.
    """

    def __init__(self, input_width: int, input_height: int, camera_intrinsics: CameraIntrinsics):
        """
        Initializes the IrisDistanceCalculator object with input dimensions and camera intrinsics.

        :param input_width: The width of the input image.
        :param input_height: The height of the input image.
        :param camera_intrinsics: The camera intrinsics.
        """

        self.average_iris_diameter = 11.7  # ranges between 10.2 - 13.0

        self.input_width = input_width
        self.input_height = input_height
        self.camera_intrinsics = camera_intrinsics

    def setup(self):
        """
        Setup method to initialize the node before processing.
        """

        pass

    def process(self, faces: ResultList[BlazeFaceMesh]) -> ResultList[IrisDistanceResult]:
        """
        Processes a list of face detection results and calculates the iris distance.

        :param faces: A list of BlazeFaceMesh objects representing face detections.

        :return: A list of IrisDistanceResult objects containing iris distances.
        """

        results = ResultList()
        for face in faces:
            right_iris = self.measure_iris_distance_in_m(face, BlazeFaceMesh.RIGHT_IRIS_INDICES)
            left_iris = self.measure_iris_distance_in_m(face, BlazeFaceMesh.LEFT_IRIS_INDICES)

            iris_distance_result = IrisDistanceResult(right_iris, left_iris)
            results.append(iris_distance_result)

        return results

    def measure_iris_distance_in_m(self, face: BlazeFaceMesh, iris_indices: Sequence[int]) -> IrisParameter:
        """
        Measures the distance between iris in a face and calculates the corresponding z-distance.

        :param face: The face detection result.
        :param iris_indices: A list of indices representing the iris landmarks.

        :return: An object containing the average z-distance, iris size, and position.
        """

        iris_landmarks = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in iris_indices])
        iris_landmarks *= np.array([self.input_width, self.input_height], dtype=float)

        iris_distances = cdist(iris_landmarks, iris_landmarks, metric='euclidean')
        max_size = float(iris_distances.max())

        fx_z = self._calculate_z_distance_in_m(max_size, self.camera_intrinsics.fx)
        fy_z = self._calculate_z_distance_in_m(max_size, self.camera_intrinsics.fy)

        average_z_in_m = float(np.mean([fx_z, fy_z]))

        # used only for result
        position = landmarks_center_by_indices(face.landmarks, iris_indices)

        return IrisParameter(average_z_in_m, max_size, position)

    def _calculate_z_distance_in_m(self, iris_size: float, focal_parameter: float) -> float:
        """
        Calculates the z-distance between the iris and the camera.

        :param iris_size: The size of the iris.
        :param focal_parameter: The focal parameter of the camera.

        :return: The calculated z-distance in meters.
        """

        return (focal_parameter * (self.average_iris_diameter / iris_size)) / 1000.0

    def release(self):
        """
        Releases any resources used by the node.
        """

        pass

    def configure(self, args: Namespace):
        """
        Configures the node with command-line arguments.

        :param args: The command-line arguments.
        """

        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argument parser for configuration.
        """

        pass
