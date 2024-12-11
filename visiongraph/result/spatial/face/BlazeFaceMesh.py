from typing import Optional, Sequence, List, Tuple

import mediapipe as mp
import numpy as np
import vector

from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult
from visiongraph.util.VectorUtils import landmarks_center_by_indices

_mp_face_mesh = mp.solutions.face_mesh


class BlazeFaceMesh(FaceLandmarkResult):
    # more information about the indices:
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

    LEFT_EYE_CENTER_INDICES = frozenset([386, 374])
    RIGHT_EYE_CENTER_INDICES = frozenset([159, 145])

    LEFT_IRIS_INDICES = frozenset([474, 475, 476, 477])
    RIGHT_IRIS_INDICES = frozenset([469, 470, 471, 472])

    LEFT_EYE_BOX_INDICES = frozenset([*LEFT_EYE_CENTER_INDICES, 362, 263])
    RIGHT_EYE_BOX_INDICES = frozenset([*RIGHT_EYE_CENTER_INDICES, 33, 133])

    # lips indices
    UPPER_VERMILION_TOP_INDICES = frozenset([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291])
    UPPER_VERMILION_BOTTOM_INDICES = frozenset([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308])

    UPPER_VERMILION_INDICES = frozenset([*UPPER_VERMILION_TOP_INDICES, *UPPER_VERMILION_BOTTOM_INDICES])

    LOWER_VERMILION_TOP_INDICES = frozenset([95, 88, 178, 87, 14, 317, 402, 318, 324])
    LOWER_VERMILION_BOTTOM_INDICES = frozenset([146, 91, 181, 84, 17, 314, 405, 321, 375])

    LOWER_VERMILION_INDICES = frozenset([*LOWER_VERMILION_TOP_INDICES, *LOWER_VERMILION_BOTTOM_INDICES])

    LIPS_INDICES = frozenset([*UPPER_VERMILION_INDICES, *LOWER_VERMILION_INDICES])

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
