from typing import Optional, List, Tuple, FrozenSet, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

COCO_CONNECTIONS = frozenset([
    (0, 1),  # nose → left eye
    (0, 2),  # nose → right eye
    (1, 3),  # left eye → left ear
    (2, 4),  # right eye → right ear
    # (0, 5),  # nose → left shoulder
    # (0, 6),  # nose → right shoulder
    (5, 6),  # left shoulder → right shoulder
    (5, 7),  # left shoulder → left elbow
    (7, 9),  # left elbow → left wrist
    (6, 8),  # right shoulder → right elbow
    (8, 10),  # right elbow → right wrist
    (11, 12),  # left hip → right hip
    (5, 11),  # left shoulder → left hip
    (11, 13),  # left hip → left knee
    (13, 15),  # left knee → left ankle
    (6, 12),  # right shoulder → right hip
    (12, 14),  # right hip → right knee
    (14, 16),  # right knee → right ankle
])


class COCOPose(PoseLandmarkResult):
    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return COCO_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def left_ear(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def right_ear(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def left_elbow(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def right_elbow(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def left_wrist(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def right_wrist(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def left_hip(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def right_hip(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def left_knee(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def right_knee(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def left_ankle(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def right_ankle(self) -> vector.Vector4D:
        return self.landmarks[16]
