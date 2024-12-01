from dataclasses import dataclass

import cv2
import numpy as np
import vector
from vector import Vector4D

from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.VectorUtils import lerp_vector_4d


@dataclass
class IrisParameter:
    """
    Represents iris parameters with distance, diameter and position.
    """
    distance: float
    diameter: float
    position: vector.Vector4D


class IrisDistanceResult(BaseResult):
    """
    A result class for calculating the distance and center of irises in an image.
    """

    def __init__(self, right_iris: IrisParameter, left_iris: IrisParameter):
        """
        Initializes the IrisDistanceResult object with the given irises.

        :param right_iris: The parameters of the right iris.
        :param left_iris: The parameters of the left iris.
        """
        self.right_iris: IrisParameter = right_iris
        self.left_iris: IrisParameter = left_iris

    def average_iris_distance(self) -> float:
        """
        Calculates and returns the average distance between the camera and both irises.

        :return: The average distance as a float.
        """
        return float(np.mean([self.right_iris.distance, self.left_iris.distance]))

    def head_center(self) -> Vector4D:
        """
        Calculates and returns the center position of the face by interpolating between the positions of the left and right irises.

        :return: The interpolated 3D position as a vector.
        """
        return lerp_vector_4d(self.left_iris.position, self.right_iris.position, 0.5)

    @staticmethod
    def _mark_point(image: np.ndarray, point: vector.Vector2D, radius: float, w: float, h: float):
        """
        Draws a circle at the given position on the image with the specified radius.

        :param image: The input image.
        :param point: The 2D position to draw the circle at.
        :param radius: The radius of the circle.
        :param w: The width of the image.
        :param h: The height of the image.
        """
        x = int(point.x * w)
        y = int(point.y * h)

        cv2.circle(image, (x, y), round(radius), (0, 255, 255), 1)

    def annotate(self, image: np.ndarray, **kwargs) -> None:
        """
        Annotates the given image with the positions of both irises.

        :param image: The input image.
        """
        h, w = image.shape[:2]

        # self.face.annotate(image, **kwargs)
        self._mark_point(image, self.left_iris.position.to_xy(), self.left_iris.diameter / 2, w, h)
        self._mark_point(image, self.right_iris.position.to_xy(), self.left_iris.diameter / 2, w, h)
