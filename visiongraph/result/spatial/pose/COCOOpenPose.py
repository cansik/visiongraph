from typing import FrozenSet, Tuple, List

import vector

from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util.VectorUtils import list_of_vector4D

COCO_OPEN_POSE_PAIRS = frozenset([
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
    (0, 14), (14, 16), (0, 15), (15, 17)
])
"""
Frozen set representing pairs of keypoints that define connections in the COCO OpenPose model.
"""

COCO_REORDER_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
"""
Mapping for reordering keypoints from the COCO OpenPose output format.
"""

COCO_OPEN_POSE_KEYPOINT_COUNT = 18
"""
Total number of keypoints defined in the COCO OpenPose model.
"""


class COCOOpenPose(PoseLandmarkResult):
    """
    Class representing the COCO OpenPose model with methods to access various body landmarks.
    """

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Gets the connections between keypoints defined in COCO OpenPose.

        Returns:
            FrozenSet[Tuple[int, int]]: A frozen set of tuples indicating the connections between keypoints.
        """
        return COCO_OPEN_POSE_PAIRS

    @property
    def nose(self) -> vector.Vector4D:
        """
        Gets the position of the nose landmark.

        Returns:
            vector.Vector4D: The nose landmark's position and score.
        """
        return self.landmarks[0]

    @property
    def neck(self) -> vector.Vector4D:
        """
        Gets the position of the neck landmark.

        Returns:
            vector.Vector4D: The neck landmark's position and score.
        """
        return self.landmarks[1]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Gets the position of the right shoulder landmark.

        Returns:
            vector.Vector4D: The right shoulder landmark's position and score.
        """
        return self.landmarks[2]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Gets the position of the right elbow landmark.

        Returns:
            vector.Vector4D: The right elbow landmark's position and score.
        """
        return self.landmarks[3]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Gets the position of the right wrist landmark.

        Returns:
            vector.Vector4D: The right wrist landmark's position and score.
        """
        return self.landmarks[4]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Gets the position of the left shoulder landmark.

        Returns:
            vector.Vector4D: The left shoulder landmark's position and score.
        """
        return self.landmarks[5]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Gets the position of the left elbow landmark.

        Returns:
            vector.Vector4D: The left elbow landmark's position and score.
        """
        return self.landmarks[6]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Gets the position of the left wrist landmark.

        Returns:
            vector.Vector4D: The left wrist landmark's position and score.
        """
        return self.landmarks[7]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Gets the position of the right hip landmark.

        Returns:
            vector.Vector4D: The right hip landmark's position and score.
        """
        return self.landmarks[8]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Gets the position of the right knee landmark.

        Returns:
            vector.Vector4D: The right knee landmark's position and score.
        """
        return self.landmarks[9]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Gets the position of the right ankle landmark.

        Returns:
            vector.Vector4D: The right ankle landmark's position and score.
        """
        return self.landmarks[10]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Gets the position of the left hip landmark.

        Returns:
            vector.Vector4D: The left hip landmark's position and score.
        """
        return self.landmarks[11]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Gets the position of the left knee landmark.

        Returns:
            vector.Vector4D: The left knee landmark's position and score.
        """
        return self.landmarks[12]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Gets the position of the left ankle landmark.

        Returns:
            vector.Vector4D: The left ankle landmark's position and score.
        """
        return self.landmarks[13]

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Gets the position of the right eye landmark.

        Returns:
            vector.Vector4D: The right eye landmark's position and score.
        """
        return self.landmarks[14]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Gets the position of the left eye landmark.

        Returns:
            vector.Vector4D: The left eye landmark's position and score.
        """
        return self.landmarks[15]

    @property
    def right_ear(self) -> vector.Vector4D:
        """
        Gets the position of the right ear landmark.

        Returns:
            vector.Vector4D: The right ear landmark's position and score.
        """
        return self.landmarks[16]

    @property
    def left_ear(self) -> vector.Vector4D:
        """
        Gets the position of the left ear landmark.

        Returns:
            vector.Vector4D: The left ear landmark's position and score.
        """
        return self.landmarks[17]

    def to_coco_pose(self) -> COCOPose:
        """
        Converts the COCO OpenPose landmark representation to a COCOPose instance.

        Returns:
            COCOPose: An instance of COCOPose representing the landmarks and score.
        """
        # todo: fix this conversion
        coco_landmarks: List[Tuple[float, float, float, float]] = []
        for i in COCO_REORDER_MAP:
            if i < 0:
                continue

            lm = self.landmarks[i]
            coco_landmarks.append((lm.x, lm.y, lm.z, lm.t))

        return COCOPose(self.score, list_of_vector4D(coco_landmarks))
