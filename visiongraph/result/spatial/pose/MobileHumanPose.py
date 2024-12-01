from typing import FrozenSet, Tuple

import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

MOBILE_HUMAN_POSE_CONNECTIONS = frozenset([
    (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20),
    (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18)
])
"""
A frozen set of tuples representing the connections between key points
of a mobile human pose model.
"""


class MobileHumanPose(PoseLandmarkResult):
    """
    A class that represents a mobile human pose model and provides access
    to specific landmark positions in a 3D space.
    """

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Provides the connections between key points of the mobile human pose.

        :return: A set of tuples representing the
        """
        return MOBILE_HUMAN_POSE_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the nose landmark.

        :return: The position of the nose in the format
        """
        return self.landmarks[16]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left eye landmark.

        :return: The position of the left eye in the format
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right eye landmark.

        :return: The position of the right eye in the format
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left shoulder landmark.

        :return: The position of the left shoulder in the format
        """
        return self.landmarks[5]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right shoulder landmark.

        :return: The position of the right shoulder in the format
        """
        return self.landmarks[2]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left elbow landmark.

        :return: The position of the left elbow in the format
        """
        return self.landmarks[6]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right elbow landmark.

        :return: The position of the right elbow in the format
        """
        return self.landmarks[3]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left wrist landmark.

        :return: The position of the left wrist in the format
        """
        return self.landmarks[7]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right wrist landmark.

        :return: The position of the right wrist in the format
        """
        return self.landmarks[4]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left hip landmark.

        :return: The position of the left hip in the format
        """
        return self.landmarks[11]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right hip landmark.

        :return: The position of the right hip in the format
        """
        return self.landmarks[8]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left knee landmark.

        :return: The position of the left knee in the format
        """
        return self.landmarks[12]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right knee landmark.

        :return: The position of the right knee in the format
        """
        return self.landmarks[9]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left ankle landmark.

        :return: The position of the left ankle in the format
        """
        return self.landmarks[13]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right ankle landmark.

        :return: The position of the right ankle in the format
        """
        return self.landmarks[10]

    @property
    def head_top(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the top of the head landmark.

        :return: The position of the head top in the format
        """
        return self.landmarks[0]

    @property
    def thorax(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the thorax landmark.

        :return: The position of the thorax in the format
        """
        return self.landmarks[1]

    @property
    def pelvis(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the pelvis landmark.

        :return: The position of the pelvis in the format
        """
        return self.landmarks[14]

    @property
    def spine(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the spine landmark.

        :return: The position of the spine in the format
        """
        return self.landmarks[15]

    @property
    def right_hand(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right hand landmark.

        :return: The position of the right hand in the format
        """
        return self.landmarks[17]

    @property
    def left_hand(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left hand landmark.

        :return: The position of the left hand in the format
        """
        return self.landmarks[18]

    @property
    def right_toe(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right toe landmark.

        :return: The position of the right toe in the format
        """
        return self.landmarks[19]

    @property
    def left_toe(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left toe landmark.

        :return: The position of the left toe in the format
        """
        return self.landmarks[20]
