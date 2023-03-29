from dataclasses import dataclass

import cv2
import numpy as np
import vector
from vector import Vector4D

from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.VectorUtils import lerp_vector_2d, lerp_vector_4d


@dataclass
class IrisParameter:
    distance: float
    diameter: float
    position: vector.Vector4D


class IrisDistanceResult(BaseResult):

    def __init__(self, right_iris: IrisParameter, left_iris: IrisParameter):
        self.right_iris: IrisParameter = right_iris
        self.left_iris: IrisParameter = left_iris

    def average_iris_distance(self) -> float:
        return float(np.mean([self.right_iris.distance, self.left_iris.distance]))

    def head_center(self) -> Vector4D:
        return lerp_vector_4d(self.left_iris.position, self.right_iris.position, 0.5)

    @staticmethod
    def _mark_point(image: np.ndarray, point: vector.Vector2D, radius: float, w: float, h: float):
        x = int(point.x * w)
        y = int(point.y * h)

        cv2.circle(image, (x, y), round(radius), (0, 255, 255), 1)

    def annotate(self, image: np.ndarray, **kwargs):
        h, w = image.shape[:2]

        # self.face.annotate(image, **kwargs)
        self._mark_point(image, self.left_iris.position.to_xy(), self.left_iris.diameter / 2, w, h)
        self._mark_point(image, self.right_iris.position.to_xy(), self.left_iris.diameter / 2, w, h)
