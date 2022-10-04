from typing import Tuple, FrozenSet

import mediapipe as mp
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult


class BlazePose(PoseLandmarkResult):
    @property
    def connections(self) -> FrozenSet[Tuple[int, int]]:
        return mp.solutions.pose.POSE_CONNECTIONS

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    def left_eye_inner(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    def left_eye_outer(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    def right_eye_inner(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[5]

    @property
    def right_eye_outer(self) -> vector.Vector4D:
        return self.landmarks[6]

    @property
    def left_ear(self) -> vector.Vector4D:
        return self.landmarks[7]

    @property
    def right_ear(self) -> vector.Vector4D:
        return self.landmarks[8]

    @property
    def mouth_left(self) -> vector.Vector4D:
        return self.landmarks[9]

    @property
    def mouth_right(self) -> vector.Vector4D:
        return self.landmarks[10]

    @property
    def left_shoulder(self) -> vector.Vector4D:
        return self.landmarks[11]

    @property
    def right_shoulder(self) -> vector.Vector4D:
        return self.landmarks[12]

    @property
    def left_elbow(self) -> vector.Vector4D:
        return self.landmarks[13]

    @property
    def right_elbow(self) -> vector.Vector4D:
        return self.landmarks[14]

    @property
    def left_wrist(self) -> vector.Vector4D:
        return self.landmarks[15]

    @property
    def right_wrist(self) -> vector.Vector4D:
        return self.landmarks[16]

    @property
    def left_pinky(self) -> vector.Vector4D:
        return self.landmarks[17]

    @property
    def right_pinky(self) -> vector.Vector4D:
        return self.landmarks[18]

    @property
    def left_index(self) -> vector.Vector4D:
        return self.landmarks[19]

    @property
    def right_index(self) -> vector.Vector4D:
        return self.landmarks[20]

    @property
    def left_thumb(self) -> vector.Vector4D:
        return self.landmarks[21]

    @property
    def right_thumb(self) -> vector.Vector4D:
        return self.landmarks[22]

    @property
    def left_hip(self) -> vector.Vector4D:
        return self.landmarks[23]

    @property
    def right_hip(self) -> vector.Vector4D:
        return self.landmarks[24]

    @property
    def left_knee(self) -> vector.Vector4D:
        return self.landmarks[25]

    @property
    def right_knee(self) -> vector.Vector4D:
        return self.landmarks[26]

    @property
    def left_ankle(self) -> vector.Vector4D:
        return self.landmarks[27]

    @property
    def right_ankle(self) -> vector.Vector4D:
        return self.landmarks[28]

    @property
    def left_heel(self) -> vector.Vector4D:
        return self.landmarks[29]

    @property
    def right_heel(self) -> vector.Vector4D:
        return self.landmarks[30]

    @property
    def left_food_index(self) -> vector.Vector4D:
        return self.landmarks[31]

    @property
    def right_foot_index(self) -> vector.Vector4D:
        return self.landmarks[32]
