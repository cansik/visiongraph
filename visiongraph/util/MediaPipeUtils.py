from typing import Tuple

import numpy as np
import vector

from visiongraph.util.VectorUtils import list_of_vector4D


def mediapipe_landmarks_to_vector4d(mediapipe_landmarks) -> vector.VectorNumpy4D:
    raw_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in mediapipe_landmarks]
    landmarks = list_of_vector4D(raw_landmarks)
    return landmarks


def mediapipe_landmarks_to_score_and_vector4d(mediapipe_landmarks) -> Tuple[float, vector.VectorNumpy4D]:
    landmarks = mediapipe_landmarks_to_vector4d(mediapipe_landmarks)
    score = float(np.average(landmarks["t"]))
    return score, landmarks
