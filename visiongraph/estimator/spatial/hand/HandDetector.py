from abc import ABC, abstractmethod
from typing import List

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.spatial.hand.HandDetectionResult import HandDetectionResult


class FaceDetector(ObjectDetector, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> List[HandDetectionResult]:
        pass
