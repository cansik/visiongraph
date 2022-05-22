from typing import Optional, Sequence, FrozenSet, Tuple, List

import numpy as np
import vector

from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util.VectorUtils import list_of_vector4D

COCO_OPEN_POSE_PAIRS = frozenset([
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16), (0, 15), (15, 17)
])

COCO_REORDER_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

COCO_OPEN_POSE_KEYPOINT_COUNT = 18


class COCOOpenPose(PoseLandmarkResult):
    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return COCO_OPEN_POSE_PAIRS

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[0]

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
    def left_shoulder(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def left_elbow(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def left_wrist(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def right_hip(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def right_knee(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def right_ankle(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def left_hip(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def left_knee(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def left_ankle(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def right_ear(self) -> vector.Vector4D:
        return self.landmarks[16]

    @property
    def left_ear(self) -> vector.Vector4D:
        return self.landmarks[17]

    def to_coco_pose(self) -> COCOPose:
        # todo: fix this conversion
        coco_landmarks: List[Tuple[float, float, float, float]] = []
        for i in COCO_REORDER_MAP:
            if i < 0:
                continue

            lm = self.landmarks[i]
            coco_landmarks.append((lm.x, lm.y, lm.z, lm.t))

        return COCOPose(self.score, list_of_vector4D(coco_landmarks))
