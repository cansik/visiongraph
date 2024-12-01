from typing import Tuple, FrozenSet

import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

COCO_CONNECTIONS = frozenset([
    (0, 1),  # nose → left eye
    (0, 2),  # nose → right eye
    (1, 3),  # left eye → left ear
    (2, 4),  # right eye → right ear
    # (0, 5),  # nose → left shoulder
    # (0, 6),  # nose → right shoulder
    (5, 6),  # left shoulder → right shoulder
    (5, 7),  # left shoulder → left elbow
    (7, 9),  # left elbow → left wrist
    (6, 8),  # right shoulder → right elbow
    (8, 10),  # right elbow → right wrist
    (11, 12),  # left hip → right hip
    (5, 11),  # left shoulder → left hip
    (11, 13),  # left hip → left knee
    (13, 15),  # left knee → left ankle
    (6, 12),  # right shoulder → right hip
    (12, 14),  # right hip → right knee
    (14, 16),  # right knee → right ankle
])


class COCOPose(PoseLandmarkResult):
    """
    A class to represent the pose results in the COCO (Common Objects in Context) format,
    inheriting from PoseLandmarkResult. This class provides properties to access specific
    pose landmarks such as nose, eyes, shoulders, etc.
    """

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Retrieves the predefined connections between landmarks based on COCO keypoints.

        :return: A set of tuples representing the connections
        """
        return COCO_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the nose landmark.

        :return: The Vector4D representation of the nose landmark.
        """
        return self.landmarks[0]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left eye landmark.

        :return: The Vector4D representation of the left eye landmark.
        """
        return self.landmarks[1]

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right eye landmark.

        :return: The Vector4D representation of the right eye landmark.
        """
        return self.landmarks[2]

    @property
    def left_ear(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left ear landmark.

        :return: The Vector4D representation of the left ear landmark.
        """
        return self.landmarks[3]

    @property
    def right_ear(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right ear landmark.

        :return: The Vector4D representation of the right ear landmark.
        """
        return self.landmarks[4]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left shoulder landmark.

        :return: The Vector4D representation of the left shoulder landmark.
        """
        return self.landmarks[5]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right shoulder landmark.

        :return: The Vector4D representation of the right shoulder landmark.
        """
        return self.landmarks[6]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left elbow landmark.

        :return: The Vector4D representation of the left elbow landmark.
        """
        return self.landmarks[7]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right elbow landmark.

        :return: The Vector4D representation of the right elbow landmark.
        """
        return self.landmarks[8]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left wrist landmark.

        :return: The Vector4D representation of the left wrist landmark.
        """
        return self.landmarks[9]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right wrist landmark.

        :return: The Vector4D representation of the right wrist landmark.
        """
        return self.landmarks[10]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left hip landmark.

        :return: The Vector4D representation of the left hip landmark.
        """
        return self.landmarks[11]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right hip landmark.

        :return: The Vector4D representation of the right hip landmark.
        """
        return self.landmarks[12]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left knee landmark.

        :return: The Vector4D representation of the left knee landmark.
        """
        return self.landmarks[13]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right knee landmark.

        :return: The Vector4D representation of the right knee landmark.
        """
        return self.landmarks[14]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the left ankle landmark.

        :return: The Vector4D representation of the left ankle landmark.
        """
        return self.landmarks[15]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Retrieves the 3D position and score for the right ankle landmark.

        :return: The Vector4D representation of the right ankle landmark.
        """
        return self.landmarks[16]
