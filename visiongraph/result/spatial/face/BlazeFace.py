import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class BlazeFace(FaceLandmarkResult):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: BoundingBox2D):
        super().__init__(score, landmarks, bounding_box)

    @property
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def mouth_center(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def right_ear_tragion(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def left_ear_tragion(self) -> vector.Vector4D:
        return self.landmarks[5]
