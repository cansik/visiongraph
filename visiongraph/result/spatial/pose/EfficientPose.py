from typing import FrozenSet, Tuple

import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util import VectorUtils

EFFICIENT_POSE_PAIRS = frozenset([
    (0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10),
    (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)
])
"""
A set of efficient pairs of landmark indices for pose connections.
"""


class EfficientPose(PoseLandmarkResult):
    """
    A class to represent an efficient pose estimation model,
    extending the PoseLandmarkResult to provide specific landmark properties.

    Inherits:
        PoseLandmarkResult: Base class for pose landmark results.
    """

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Returns the frozen set of connections representing efficient pairs of pose landmarks.

        :return: Pairs of indices representing connections in the pose.
        """
        return EFFICIENT_POSE_PAIRS

    @property
    def head_top(self) -> vector.Vector4D:
        """
        Returns the position of the head top landmark.

        :return: The vector representing the head top position.
        """
        return self.landmarks[0]

    @property
    def nose(self) -> vector.Vector4D:
        """
        Returns the interpolated position of the nose landmark, computed as the midpoint
        between the head top and neck landmarks.

        :return: The vector representing the nose position.
        """
        return VectorUtils.lerp_vector_4d(self.head_top, self.neck, 0.5)

    @property
    def neck(self) -> vector.Vector4D:
        """
        Returns the position of the neck landmark.

        :return: The vector representing the neck position.
        """
        return self.landmarks[1]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Returns the position of the right shoulder landmark.

        :return: The vector representing the right shoulder position.
        """
        return self.landmarks[2]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Returns the position of the right elbow landmark.

        :return: The vector representing the right elbow position.
        """
        return self.landmarks[3]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Returns the position of the right wrist landmark.

        :return: The vector representing the right wrist position.
        """
        return self.landmarks[4]

    @property
    def thorax(self) -> vector.Vector4D:
        """
        Returns the position of the thorax landmark.

        :return: The vector representing the thorax position.
        """
        return self.landmarks[5]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Returns the position of the left shoulder landmark.

        :return: The vector representing the left shoulder position.
        """
        return self.landmarks[6]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Returns the position of the left elbow landmark.

        :return: The vector representing the left elbow position.
        """
        return self.landmarks[7]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Returns the position of the left wrist landmark.

        :return: The vector representing the left wrist position.
        """
        return self.landmarks[8]

    @property
    def pelvis(self) -> vector.Vector4D:
        """
        Returns the position of the pelvis landmark.

        :return: The vector representing the pelvis position.
        """
        return self.landmarks[9]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Returns the position of the right hip landmark.

        :return: The vector representing the right hip position.
        """
        return self.landmarks[10]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Returns the position of the right knee landmark.

        :return: The vector representing the right knee position.
        """
        return self.landmarks[11]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Returns the position of the right ankle landmark.

        :return: The vector representing the right ankle position.
        """
        return self.landmarks[12]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Returns the position of the left hip landmark.

        :return: The vector representing the left hip position.
        """
        return self.landmarks[13]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Returns the position of the left knee landmark.

        :return: The vector representing the left knee position.
        """
        return self.landmarks[14]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Returns the position of the left ankle landmark.

        :return: The vector representing the left ankle position.
        """
        return self.landmarks[15]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Returns a placeholder vector representing the position of the left eye landmark.

        :return: The vector with default values representing the left eye position.
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Returns a placeholder vector representing the position of the right eye landmark.

        :return: The vector with default values representing the right eye position.
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)
