from abc import abstractmethod, ABC

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.spatial.hand.HandLandmarkResult import HandLandmarkResult


class HandLandmarkEstimator(LandmarkEstimator, RoiEstimator, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> HandLandmarkResult:
        pass
