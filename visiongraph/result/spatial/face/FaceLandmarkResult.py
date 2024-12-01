from abc import ABC, abstractmethod
from typing import Optional

import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult
from visiongraph.result.spatial.face.FaceDetectionResult import FACE_DETECTION_LABEL, FACE_DETECTION_ID


class FaceLandmarkResult(LandmarkDetectionResult, ABC):
    """
    Abstract base class for face landmark detection results.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: Optional[BoundingBox2D] = None):
        """
        Initializes the FaceLandmarkResult object with a score and detected landmarks.

        :param score: The confidence of the detection.
        :param landmarks: The coordinates of the detected landmarks.
        :param bounding_box: The bounding box of the face, if available. Defaults to None.
        """
        super().__init__(FACE_DETECTION_ID, FACE_DETECTION_LABEL, score, landmarks, bounding_box)

    @property
    @abstractmethod
    def nose(self) -> vector.Vector4D:
        """
        Gets the position of the nose landmark.

        :return: The coordinates of the nose landmark.
        """
        pass

    @property
    @abstractmethod
    def left_eye(self) -> vector.Vector4D:
        """
        Gets the position of the left eye landmark.

        :return: The coordinates of the left eye landmark.
        """
        pass

    @property
    @abstractmethod
    def right_eye(self) -> vector.Vector4D:
        """
        Gets the position of the right eye landmark.

        :return: The coordinates of the right eye landmark.
        """
        pass
