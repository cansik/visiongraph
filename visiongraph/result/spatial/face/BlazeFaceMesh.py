from typing import Optional, Sequence, List, Tuple

import numpy as np
import vector
import mediapipe as mp

from visiongraph import FaceLandmarkResult
from visiongraph.util.VectorUtils import landmarks_center_by_indices

_mp_face_mesh = mp.solutions.face_mesh


class BlazeFaceMesh(FaceLandmarkResult):
    LEFT_EYE_CENTER_INDICES = [386, 374]
    RIGHT_EYE_CENTER_INDICES = [159, 145]

    LEFT_IRIS_INDICES = [474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [469, 470, 471, 472]

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(score, landmarks)

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0,
                 connections: Optional[List[Tuple[int, int]]] = _mp_face_mesh.FACEMESH_FACE_OVAL,
                 marker_size: int = 1,
                 stroke_width: int = 1, **kwargs):
        super().annotate(image, show_info, info_text, color, show_bounding_box,
                         min_score, connections, stroke_width, marker_size)

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    def left_eye(self) -> vector.Vector4D:
        return landmarks_center_by_indices(self.landmarks, self.LEFT_EYE_CENTER_INDICES)

    @property
    def right_eye(self) -> vector.Vector4D:
        return landmarks_center_by_indices(self.landmarks, self.RIGHT_EYE_CENTER_INDICES)

    @property
    def left_iris(self) -> vector.Vector4D:
        return landmarks_center_by_indices(self.landmarks, self.LEFT_IRIS_INDICES)

    @property
    def right_iris(self) -> vector.Vector4D:
        return landmarks_center_by_indices(self.landmarks, self.RIGHT_IRIS_INDICES)

    @property
    def mouth_left(self) -> vector.Vector4D:
        return self.landmarks[306]

    @property
    def mouth_right(self) -> vector.Vector4D:
        return self.landmarks[76]
