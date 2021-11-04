import vector

from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class RegressionRetailFace(FaceLandmarkResult):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(score, landmarks)

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
