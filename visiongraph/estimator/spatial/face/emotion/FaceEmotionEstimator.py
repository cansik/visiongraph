from abc import abstractmethod

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.spatial.face.EmotionClassificationResult import EmotionClassificationResult


class FaceEmotionEstimator(RoiEstimator, BaseClassifier):
    """
    An estimator for facial emotion classification.

    This class extends both RoiEstimator and BaseClassifier to provide a specific implementation for face emotion classification.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> EmotionClassificationResult:
        """
        Processes the given image to classify its face emotions.

        :param image: The input image to be processed.

        :return: The result of emotion classification.
        """
        pass
