from abc import ABC, abstractmethod

import vector

from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult
from visiongraph.result.spatial.hand.HandDetectionResult import HAND_DETECTION_ID, HAND_DETECTION_LABEL
from visiongraph.result.spatial.hand.Handedness import Handedness


class HandLandmarkResult(LandmarkDetectionResult, ABC):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, handedness: Handedness):
        super().__init__(HAND_DETECTION_ID, HAND_DETECTION_LABEL, score, landmarks)
        self.handedness = handedness

    @property
    @abstractmethod
    def wrist(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def thumb_cmc(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def thumb_mcp(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def thumb_ip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def thumb_tip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def index_finger_cmc(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def index_finger_mcp(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def index_finger_ip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def index_finger_tip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def middle_finger_cmc(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def middle_finger_mcp(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def middle_finger_ip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def middle_finger_tip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def ring_finger_cmc(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def ring_finger_mcp(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def ring_finger_ip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def ring_finger_tip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def pinky_cmc(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def pinky_mcp(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def pinky_ip(self) -> vector.Vector4D:
        pass

    @property
    @abstractmethod
    def pinky_tip(self) -> vector.Vector4D:
        pass
