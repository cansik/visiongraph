from abc import ABC, abstractmethod
from typing import Optional

import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult


class FaceLandmarkResult(LandmarkDetectionResult, ABC):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: Optional[BoundingBox2D] = None):
        super().__init__(0, "face", score, landmarks, bounding_box)

    @property
    @abstractmethod
    def nose(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_eye(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_eye(self) -> vector.Vector4D:
        pass
