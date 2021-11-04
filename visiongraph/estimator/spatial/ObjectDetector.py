from abc import ABC, abstractmethod
from typing import List

import numpy as np

from visiongraph.estimator.VisionClassifier import VisionClassifier
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class ObjectDetector(VisionClassifier, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> List[ObjectDetectionResult]:
        pass
