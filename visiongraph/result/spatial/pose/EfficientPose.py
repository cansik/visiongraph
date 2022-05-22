from typing import FrozenSet, Tuple

import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util import VectorUtils

EFFICIENT_POSE_PAIRS = frozenset([
    (0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10),
    (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)
])


class EfficientPose(PoseLandmarkResult):
    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return EFFICIENT_POSE_PAIRS

    @property
    def head_top(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def nose(self) -> vector.Vector4D:
        return VectorUtils.lerp4d(self.head_top, self.neck, 0.5)

    @property
    def neck(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def right_elbow(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def right_wrist(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def thorax(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def left_elbow(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def left_wrist(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def pelvis(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def right_hip(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def right_knee(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def right_ankle(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def left_hip(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def left_knee(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def left_ankle(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def left_eye(self) -> vector.Vector4D:
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def right_eye(self) -> vector.Vector4D:
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)
