from abc import ABC, abstractmethod
from typing import List, Tuple, FrozenSet, Optional, Sequence

import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
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
    """
    A class to represent the results of landmark detection for poses.

    Inherits from LandmarkDetectionResult and provides properties for pose landmarks.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: Optional[BoundingBox2D] = None):
        """
        Initializes the PoseLandmarkResult with a score, landmarks, and an optional bounding box.

        :param score: The score representing the confidence of the pose detection.
        :param landmarks: The detected landmarks in a 4D vector format.
        :param bounding_box: An optional bounding box around the pose.
        """
        super().__init__(POSE_DETECTION_ID, POSE_DETECTION_NAME, score, landmarks, bounding_box=bounding_box)

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, show_bounding_box: bool = False,
                 min_score: float = 0, **kwargs):
        """
        Annotates the image with the pose landmarks and optional information.

        :param image: The image to annotate.
        :param show_info: Flag to show additional information.
        :param info_text: Custom text to show on the image.
        :param color: Color for annotation.
        :param show_bounding_box: Whether to display the bounding box.
        :param min_score: Minimum score threshold for displaying landmarks.
        :param **kwargs: Additional keyword arguments for customization.
        """
        super().annotate(image, show_info, info_text, color, show_bounding_box, min_score,
                         connections=self.connections, **kwargs)

    @property
    def default_landmarks(self) -> List[vector.Vector4D]:
        """
        Retrieves the default landmarks for the pose.

        :return: A list of default pose landmarks.
        """
        return [getattr(self, lm_name) for lm_name in DEFAULT_POSE_LANDMARKS]

    @property
    @abstractmethod
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Abstract property that must return the connections between landmarks as a set of tuples.

        :return: A frozen set containing pairs of landmark indices that are connected.
        """
        pass

    @property
    @abstractmethod
    def nose(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the nose landmark.

        :return: The vector position of the nose.
        """
        pass

    @property
    @abstractmethod
    def left_eye(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left eye landmark.

        :return: The vector position of the left eye.
        """
        pass

    @property
    @abstractmethod
    def right_eye(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right eye landmark.

        :return: The vector position of the right eye.
        """
        pass

    @property
    @abstractmethod
    def left_shoulder(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left shoulder landmark.

        :return: The vector position of the left shoulder.
        """
        pass

    @property
    @abstractmethod
    def right_shoulder(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right shoulder landmark.

        :return: The vector position of the right shoulder.
        """
        pass

    @property
    @abstractmethod
    def left_elbow(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left elbow landmark.

        :return: The vector position of the left elbow.
        """
        pass

    @property
    @abstractmethod
    def right_elbow(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right elbow landmark.

        :return: The vector position of the right elbow.
        """
        pass

    @property
    @abstractmethod
    def left_wrist(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left wrist landmark.

        :return: The vector position of the left wrist.
        """
        pass

    @property
    @abstractmethod
    def right_wrist(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right wrist landmark.

        :return: The vector position of the right wrist.
        """
        pass

    @property
    @abstractmethod
    def left_hip(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left hip landmark.

        :return: The vector position of the left hip.
        """
        pass

    @property
    @abstractmethod
    def right_hip(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right hip landmark.

        :return: The vector position of the right hip.
        """
        pass

    @property
    @abstractmethod
    def left_knee(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left knee landmark.

        :return: The vector position of the left knee.
        """
        pass

    @property
    @abstractmethod
    def right_knee(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right knee landmark.

        :return: The vector position of the right knee.
        """
        pass

    @property
    @abstractmethod
    def left_ankle(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the left ankle landmark.

        :return: The vector position of the left ankle.
        """
        pass

    @property
    @abstractmethod
    def right_ankle(self) -> vector.Vector4D:
        """
        Abstract property that must return the position of the right ankle landmark.

        :return: The vector position of the right ankle.
        """
        pass
