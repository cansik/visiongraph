from typing import Tuple

import numpy as np
import vector

from visiongraph.util.VectorUtils import list_of_vector4D


def mediapipe_landmarks_to_vector4d(mediapipe_landmarks) -> vector.VectorNumpy4D:
    """
    Converts Mediapipe landmarks to a collection of 4D vectors.

    :param mediapipe_landmarks: A collection of landmarks obtained from the Mediapipe model.

    :return: A structured set of 4D vectors representing the landmarks,
    """
    raw_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in mediapipe_landmarks]
    landmarks = list_of_vector4D(raw_landmarks)
    return landmarks


def mediapipe_landmarks_to_score_and_vector4d(mediapipe_landmarks) -> Tuple[float, vector.VectorNumpy4D]:
    """
    Extracts a visibility score and converts Mediapipe landmarks to a collection of 4D vectors.

    :param mediapipe_landmarks: A collection of landmarks obtained from the Mediapipe model.

    :return: A tuple containing the average visibility score
    """
    landmarks = mediapipe_landmarks_to_vector4d(mediapipe_landmarks)
    score = float(np.average(landmarks["t"]))
    return score, landmarks
