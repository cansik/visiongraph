from abc import ABC, abstractmethod
from typing import List

import vector

from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

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
        super().__init__(0, "pose", score, landmarks)

    # todo: implement pose connections for base pose landmark

    @property
    def default_landmarks(self) -> List[vector.Vector4D]:
        return [getattr(self, lm_name) for lm_name in DEFAULT_POSE_LANDMARKS]

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
