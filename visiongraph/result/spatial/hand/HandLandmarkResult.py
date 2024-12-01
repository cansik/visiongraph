from abc import ABC, abstractmethod

import vector

from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult
from visiongraph.result.spatial.hand.HandDetectionResult import HAND_DETECTION_ID, HAND_DETECTION_LABEL
from visiongraph.result.spatial.hand.Handedness import Handedness


class HandLandmarkResult(LandmarkDetectionResult, ABC):
    """
    Abstract class representing the result of hand landmark detection.

    Inherits from LandmarkDetectionResult and includes methods to retrieve
    specific hand landmarks and handedness information.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, handedness: Handedness):
        """
        Initializes a HandLandmarkResult instance.

        :param score: The confidence score of the detection.
        :param landmarks: The detected landmarks as a 4D vector.
        :param handedness: The handedness of the detected hand (left or right).
        """
        super().__init__(HAND_DETECTION_ID, HAND_DETECTION_LABEL, score, landmarks)
        self.handedness = handedness

    @property
    @abstractmethod
    def wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the wrist landmark.

        :return: The coordinates of the wrist landmark.
        """
        pass

    @property
    @abstractmethod
    def thumb_cmc(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the thumb carpometacarpal (CMC) landmark.

        :return: The coordinates of the thumb CMC landmark.
        """
        pass

    @property
    @abstractmethod
    def thumb_mcp(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the thumb metacarpophalangeal (MCP) landmark.

        :return: The coordinates of the thumb MCP landmark.
        """
        pass

    @property
    @abstractmethod
    def thumb_ip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the thumb interphalangeal (IP) landmark.

        :return: The coordinates of the thumb IP landmark.
        """
        pass

    @property
    @abstractmethod
    def thumb_tip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the thumb tip landmark.

        :return: The coordinates of the thumb tip landmark.
        """
        pass

    @property
    @abstractmethod
    def index_finger_cmc(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the index finger carpometacarpal (CMC) landmark.

        :return: The coordinates of the index finger CMC landmark.
        """
        pass

    @property
    @abstractmethod
    def index_finger_mcp(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the index finger metacarpophalangeal (MCP) landmark.

        :return: The coordinates of the index finger MCP landmark.
        """
        pass

    @property
    @abstractmethod
    def index_finger_ip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the index finger interphalangeal (IP) landmark.

        :return: The coordinates of the index finger IP landmark.
        """
        pass

    @property
    @abstractmethod
    def index_finger_tip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the index finger tip landmark.

        :return: The coordinates of the index finger tip landmark.
        """
        pass

    @property
    @abstractmethod
    def middle_finger_cmc(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the middle finger carpometacarpal (CMC) landmark.

        :return: The coordinates of the middle finger CMC landmark.
        """
        pass

    @property
    @abstractmethod
    def middle_finger_mcp(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the middle finger metacarpophalangeal (MCP) landmark.

        :return: The coordinates of the middle finger MCP landmark.
        """
        pass

    @property
    @abstractmethod
    def middle_finger_ip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the middle finger interphalangeal (IP) landmark.

        :return: The coordinates of the middle finger IP landmark.
        """
        pass

    @property
    @abstractmethod
    def middle_finger_tip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the middle finger tip landmark.

        :return: The coordinates of the middle finger tip landmark.
        """
        pass

    @property
    @abstractmethod
    def ring_finger_cmc(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the ring finger carpometacarpal (CMC) landmark.

        :return: The coordinates of the ring finger CMC landmark.
        """
        pass

    @property
    @abstractmethod
    def ring_finger_mcp(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the ring finger metacarpophalangeal (MCP) landmark.

        :return: The coordinates of the ring finger MCP landmark.
        """
        pass

    @property
    @abstractmethod
    def ring_finger_ip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the ring finger interphalangeal (IP) landmark.

        :return: The coordinates of the ring finger IP landmark.
        """
        pass

    @property
    @abstractmethod
    def ring_finger_tip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the ring finger tip landmark.

        :return: The coordinates of the ring finger tip landmark.
        """
        pass

    @property
    @abstractmethod
    def pinky_cmc(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the pinky carpometacarpal (CMC) landmark.

        :return: The coordinates of the pinky CMC landmark.
        """
        pass

    @property
    @abstractmethod
    def pinky_mcp(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the pinky metacarpophalangeal (MCP) landmark.

        :return: The coordinates of the pinky MCP landmark.
        """
        pass

    @property
    @abstractmethod
    def pinky_ip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the pinky interphalangeal (IP) landmark.

        :return: The coordinates of the pinky IP landmark.
        """
        pass

    @property
    @abstractmethod
    def pinky_tip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position of the pinky tip landmark.

        :return: The coordinates of the pinky tip landmark.
        """
        pass
