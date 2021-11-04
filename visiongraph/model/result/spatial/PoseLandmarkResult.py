from abc import ABC, abstractmethod

import vector

from visiongraph.model.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult


class PoseLandmarkResult(LandmarkDetectionResult, ABC):
    # todo: implement pose connections for base pose landmark

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
