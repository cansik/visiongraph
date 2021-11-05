from typing import Optional

import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class RegressionFace(FaceLandmarkResult):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: Optional[BoundingBox2D] = None):
        super().__init__(score, landmarks, bounding_box)

    @property
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def mouth_left(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def mouth_right(self) -> vector.Vector4D:
        return self.landmarks[4]
