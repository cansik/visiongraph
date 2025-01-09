from typing import Optional, Sequence, List, Tuple

import mediapipe as mp
import numpy as np
import vector

from visiongraph.result.spatial.face.BlendShape import BlendShape
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult
from visiongraph.util.MathUtils import decompose_transformation_matrix
from visiongraph.util.VectorUtils import landmarks_center_by_indices

_mp_face_mesh = mp.solutions.face_mesh


class BlazeFaceMesh(FaceLandmarkResult):
    # more information about the indices:
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    NOSE_INDEX = 1

    PHILTRUM_INDEX = 164

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

    # based on BlendShape estimation network
    # https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
    FEATURES_148 = frozenset([
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
        81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
        154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
        249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312,
        314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381,
        382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472,
        473, 474, 475, 476, 477
    ])

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(score, landmarks)

        self.blend_shapes: Optional[List[BlendShape]] = None
        self.transformation_matrix: Optional[np.ndarray] = None

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0,
                 connections: Optional[List[Tuple[int, int]]] = _mp_face_mesh.FACEMESH_FACE_OVAL,
                 marker_size: int = 1,
                 stroke_width: int = 1, **kwargs):
        super().annotate(image, show_info, info_text, color, show_bounding_box,
                         min_score, connections, stroke_width, marker_size, **kwargs)

    def normalize_landmarks(self, origin_landmark_index: int = NOSE_INDEX) -> np.ndarray:
        """
        Normalize 3D landmark points to a canonical space using the inverse of
        the specified transformation matrix.

        :params origin_landmark_index: Index of the origin landmark for the normalisation.
        :returns: A NumPy array containing the normalized 3D landmarks.
        """
        inv_matrix = np.linalg.inv(self.transformation_matrix)
        inv_rotation, inv_translation, inv_scale = decompose_transformation_matrix(inv_matrix)

        vertices = np.array([[e.x, 1 - e.y, -e.z] for e in self.landmarks], dtype=np.float32)
        canonical_vertices = vertices @ inv_rotation.T

        origin = canonical_vertices[origin_landmark_index]

        # Translate vertices to center the origin
        canonical_vertices -= origin

        # Normalize vertices to be between -1 and +1 based on the maximum distance from the origin
        max_distance = np.max(np.linalg.norm(canonical_vertices, axis=1))
        normalized_vertices = canonical_vertices / max_distance

        return normalized_vertices

    @property
    def nose(self) -> vector.Vector4D:
        return self.landmarks[self.NOSE_INDEX]

    @property
    def philtrum(self) -> vector.Vector4D:
        return self.landmarks[self.PHILTRUM_INDEX]

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
