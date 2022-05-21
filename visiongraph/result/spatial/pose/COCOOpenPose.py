from typing import Optional, Sequence, FrozenSet, Tuple

import numpy as np
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

COCO_OPEN_POSE_PAIRS = frozenset([
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16), (0, 15), (15, 17), (2, 17), (5, 16)
])

COCO_OPEN_POSE_KEYPOINT_COUNT = 18


class COCOOpenPose(PoseLandmarkResult):
    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, show_bounding_box: bool = False,
                 min_score: float = 0, **kwargs):
        super().annotate(image, show_info, info_text, color, show_bounding_box, min_score,
                         connections=COCO_OPEN_POSE_PAIRS, **kwargs)

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return COCO_OPEN_POSE_PAIRS

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
