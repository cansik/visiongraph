from abc import abstractmethod

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.spatial.face.EmotionClassificationResult import EmotionClassificationResult


class FaceEmotionEstimator(RoiEstimator, BaseClassifier):
    @abstractmethod
    def process(self, image: np.ndarray) -> EmotionClassificationResult:
        pass
