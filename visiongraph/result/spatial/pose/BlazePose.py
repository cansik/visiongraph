from typing import Tuple, FrozenSet

import mediapipe as mp
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult


class BlazePose(PoseLandmarkResult):
    """
    A class to represent the BlazePose model for pose landmarks.

    Inherits from PoseLandmarkResult and provides properties to access various
    landmarks of the human body in a 3D space represented as Vector4D.
    """

    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        """
        Returns the connections between pose landmarks as a frozen set of tuples.

        Each tuple contains two integer indices representing connected landmarks.

        :return: The connections defined in MediaPipe.
        """
        return mp.solutions.pose.POSE_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        """
        Returns the position of the nose landmark.

        :return: The 3D position of the nose.
        """
        return self.landmarks[0]

    @property
    def left_eye_inner(self) -> vector.Vector4D:
        """
        Returns the position of the inner left eye landmark.

        :return: The 3D position of the inner left eye.
        """
        return self.landmarks[1]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Returns the position of the left eye landmark.

        :return: The 3D position of the left eye.
        """
        return self.landmarks[2]

    @property
    def left_eye_outer(self) -> vector.Vector4D:
        """
        Returns the position of the outer left eye landmark.

        :return: The 3D position of the outer left eye.
        """
        return self.landmarks[3]

    @property
    def right_eye_inner(self) -> vector.Vector4D:
        """
        Returns the position of the inner right eye landmark.

        :return: The 3D position of the inner right eye.
        """
        return self.landmarks[4]

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Returns the position of the right eye landmark.

        :return: The 3D position of the right eye.
        """
        return self.landmarks[5]

    @property
    def right_eye_outer(self) -> vector.Vector4D:
        """
        Returns the position of the outer right eye landmark.

        :return: The 3D position of the outer right eye.
        """
        return self.landmarks[6]

    @property
    def left_ear(self) -> vector.Vector4D:
        """
        Returns the position of the left ear landmark.

        :return: The 3D position of the left ear.
        """
        return self.landmarks[7]

    @property
    def right_ear(self) -> vector.Vector4D:
        """
        Returns the position of the right ear landmark.

        :return: The 3D position of the right ear.
        """
        return self.landmarks[8]

    @property
    def mouth_left(self) -> vector.Vector4D:
        """
        Returns the position of the left mouth corner landmark.

        :return: The 3D position of the left mouth corner.
        """
        return self.landmarks[9]

    @property
    def mouth_right(self) -> vector.Vector4D:
        """
        Returns the position of the right mouth corner landmark.

        :return: The 3D position of the right mouth corner.
        """
        return self.landmarks[10]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        """
        Returns the position of the left shoulder landmark.

        :return: The 3D position of the left shoulder.
        """
        return self.landmarks[11]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        """
        Returns the position of the right shoulder landmark.

        :return: The 3D position of the right shoulder.
        """
        return self.landmarks[12]

    @property
    def left_elbow(self) -> vector.Vector4D:
        """
        Returns the position of the left elbow landmark.

        :return: The 3D position of the left elbow.
        """
        return self.landmarks[13]

    @property
    def right_elbow(self) -> vector.Vector4D:
        """
        Returns the position of the right elbow landmark.

        :return: The 3D position of the right elbow.
        """
        return self.landmarks[14]

    @property
    def left_wrist(self) -> vector.Vector4D:
        """
        Returns the position of the left wrist landmark.

        :return: The 3D position of the left wrist.
        """
        return self.landmarks[15]

    @property
    def right_wrist(self) -> vector.Vector4D:
        """
        Returns the position of the right wrist landmark.

        :return: The 3D position of the right wrist.
        """
        return self.landmarks[16]

    @property
    def left_pinky(self) -> vector.Vector4D:
        """
        Returns the position of the left pinky landmark.

        :return: The 3D position of the left pinky.
        """
        return self.landmarks[17]

    @property
    def right_pinky(self) -> vector.Vector4D:
        """
        Returns the position of the right pinky landmark.

        :return: The 3D position of the right pinky.
        """
        return self.landmarks[18]

    @property
    def left_index(self) -> vector.Vector4D:
        """
        Returns the position of the left index finger landmark.

        :return: The 3D position of the left index finger.
        """
        return self.landmarks[19]

    @property
    def right_index(self) -> vector.Vector4D:
        """
        Returns the position of the right index finger landmark.

        :return: The 3D position of the right index finger.
        """
        return self.landmarks[20]

    @property
    def left_thumb(self) -> vector.Vector4D:
        """
        Returns the position of the left thumb landmark.

        :return: The 3D position of the left thumb.
        """
        return self.landmarks[21]

    @property
    def right_thumb(self) -> vector.Vector4D:
        """
        Returns the position of the right thumb landmark.

        :return: The 3D position of the right thumb.
        """
        return self.landmarks[22]

    @property
    def left_hip(self) -> vector.Vector4D:
        """
        Returns the position of the left hip landmark.

        :return: The 3D position of the left hip.
        """
        return self.landmarks[23]

    @property
    def right_hip(self) -> vector.Vector4D:
        """
        Returns the position of the right hip landmark.

        :return: The 3D position of the right hip.
        """
        return self.landmarks[24]

    @property
    def left_knee(self) -> vector.Vector4D:
        """
        Returns the position of the left knee landmark.

        :return: The 3D position of the left knee.
        """
        return self.landmarks[25]

    @property
    def right_knee(self) -> vector.Vector4D:
        """
        Returns the position of the right knee landmark.

        :return: The 3D position of the right knee.
        """
        return self.landmarks[26]

    @property
    def left_ankle(self) -> vector.Vector4D:
        """
        Returns the position of the left ankle landmark.

        :return: The 3D position of the left ankle.
        """
        return self.landmarks[27]

    @property
    def right_ankle(self) -> vector.Vector4D:
        """
        Returns the position of the right ankle landmark.

        :return: The 3D position of the right ankle.
        """
        return self.landmarks[28]

    @property
    def left_heel(self) -> vector.Vector4D:
        """
        Returns the position of the left heel landmark.

        :return: The 3D position of the left heel.
        """
        return self.landmarks[29]

    @property
    def right_heel(self) -> vector.Vector4D:
        """
        Returns the position of the right heel landmark.

        :return: The 3D position of the right heel.
        """
        return self.landmarks[30]

    @property
    def left_food_index(self) -> vector.Vector4D:
        """
        Returns the position of the left foot index landmark.

        :return: The 3D position of the left foot index.
        """
        return self.landmarks[31]

    @property
    def right_foot_index(self) -> vector.Vector4D:
        """
        Returns the position of the right foot index landmark.

        :return: The 3D position of the right foot index.
        """
        return self.landmarks[32]
