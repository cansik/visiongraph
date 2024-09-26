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

        Returns:
            FrozenSet[Tuple[int, int]]: A set of tuples representing the 
            connections between landmark indices.
        """
        return MOBILE_HUMAN_POSE_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the nose landmark.

        Returns:
            vector.Vector4D: The position of the nose in the format 
            (x, y, z, t).
        """
        return self.landmarks[16]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left eye landmark.

        Returns:
            vector.Vector4D: The position of the left eye in the format 
            (x, y, z, t).
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right eye landmark.

        Returns:
            vector.Vector4D: The position of the right eye in the format 
            (x, y, z, t).
        """
        return vector.obj(x=0.0, y=0.0, z=0.0, t=0.0)

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left shoulder landmark.

        Returns:
            vector.Vector4D: The position of the left shoulder in the format 
            (x, y, z, t).
        """
        return self.landmarks[5]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right shoulder landmark.

        Returns:
            vector.Vector4D: The position of the right shoulder in the format 
            (x, y, z, t).
        """
        return self.landmarks[2]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left elbow landmark.

        Returns:
            vector.Vector4D: The position of the left elbow in the format 
            (x, y, z, t).
        """
        return self.landmarks[6]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right elbow landmark.

        Returns:
            vector.Vector4D: The position of the right elbow in the format 
            (x, y, z, t).
        """
        return self.landmarks[3]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left wrist landmark.

        Returns:
            vector.Vector4D: The position of the left wrist in the format 
            (x, y, z, t).
        """
        return self.landmarks[7]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right wrist landmark.

        Returns:
            vector.Vector4D: The position of the right wrist in the format 
            (x, y, z, t).
        """
        return self.landmarks[4]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left hip landmark.

        Returns:
            vector.Vector4D: The position of the left hip in the format 
            (x, y, z, t).
        """
        return self.landmarks[11]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right hip landmark.

        Returns:
            vector.Vector4D: The position of the right hip in the format 
            (x, y, z, t).
        """
        return self.landmarks[8]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left knee landmark.

        Returns:
            vector.Vector4D: The position of the left knee in the format 
            (x, y, z, t).
        """
        return self.landmarks[12]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right knee landmark.

        Returns:
            vector.Vector4D: The position of the right knee in the format 
            (x, y, z, t).
        """
        return self.landmarks[9]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left ankle landmark.

        Returns:
            vector.Vector4D: The position of the left ankle in the format 
            (x, y, z, t).
        """
        return self.landmarks[13]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right ankle landmark.

        Returns:
            vector.Vector4D: The position of the right ankle in the format 
            (x, y, z, t).
        """
        return self.landmarks[10]

    @property
    def head_top(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the top of the head landmark.

        Returns:
            vector.Vector4D: The position of the head top in the format 
            (x, y, z, t).
        """
        return self.landmarks[0]

    @property
    def thorax(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the thorax landmark.

        Returns:
            vector.Vector4D: The position of the thorax in the format 
            (x, y, z, t).
        """
        return self.landmarks[1]

    @property
    def pelvis(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the pelvis landmark.

        Returns:
            vector.Vector4D: The position of the pelvis in the format 
            (x, y, z, t).
        """
        return self.landmarks[14]

    @property
    def spine(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the spine landmark.

        Returns:
            vector.Vector4D: The position of the spine in the format 
            (x, y, z, t).
        """
        return self.landmarks[15]

    @property
    def right_hand(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right hand landmark.

        Returns:
            vector.Vector4D: The position of the right hand in the format 
            (x, y, z, t).
        """
        return self.landmarks[17]

    @property
    def left_hand(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left hand landmark.

        Returns:
            vector.Vector4D: The position of the left hand in the format 
            (x, y, z, t).
        """
        return self.landmarks[18]

    @property
    def right_toe(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the right toe landmark.

        Returns:
            vector.Vector4D: The position of the right toe in the format 
            (x, y, z, t).
        """
        return self.landmarks[19]

    @property
    def left_toe(self) -> vector.Vector4D:
        """
        Retrieves the 3D position vector of the left toe landmark.

        Returns:
            vector.Vector4D: The position of the left toe in the format 
            (x, y, z, t).
        """
        return self.landmarks[20]
