from abc import ABC, abstractmethod
from typing import List, Tuple, FrozenSet, Optional, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

POSE_DETECTION_ID = 0
POSE_DETECTION_NAME = "pose"

DEFAULT_POSE_LANDMARKS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


class PoseLandmarkResult(LandmarkDetectionResult, ABC):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(POSE_DETECTION_ID, POSE_DETECTION_NAME, score, landmarks)

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, show_bounding_box: bool = False,
                 min_score: float = 0, **kwargs):
        super().annotate(image, show_info, info_text, color, show_bounding_box, min_score,
                         connections=self.connections, **kwargs)

    @property
    def default_landmarks(self) -> List[vector.Vector4D]:
        return [getattr(self, lm_name) for lm_name in DEFAULT_POSE_LANDMARKS]

    @property
    @abstractmethod
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        pass

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

    @property
    @abstractmethod
    def left_shoulder(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_shoulder(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_elbow(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_elbow(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_wrist(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_wrist(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_hip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_hip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_knee(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_knee(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def left_ankle(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def right_ankle(self) -> vector.Vector4D:
        pass
