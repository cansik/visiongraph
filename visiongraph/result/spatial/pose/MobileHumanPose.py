from typing import Optional, Sequence, FrozenSet, Tuple

import numpy as np
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

MOBILE_HUMAN_POSE_CONNECTIONS = frozenset([
    (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
    (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18)
])


class MobileHumanPose(PoseLandmarkResult):
    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return MOBILE_HUMAN_POSE_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[16]

    @property
    def left_eye(self) -> vector.Vector4D:
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def right_eye(self) -> vector.Vector4D:
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def left_shoulder(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def left_elbow(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def right_elbow(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def left_wrist(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def right_wrist(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def left_hip(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def right_hip(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def left_knee(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def right_knee(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def left_ankle(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def right_ankle(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def head_top(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def thorax(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def pelvis(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def spine(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def right_hand(self) -> vector.Vector4D:
        return self.landmarks[17]

    @property
    def left_hand(self) -> vector.Vector4D:
        return self.landmarks[18]

    @property
    def right_toe(self) -> vector.Vector4D:
        return self.landmarks[19]

    @property
    def left_toe(self) -> vector.Vector4D:
        return self.landmarks[20]
