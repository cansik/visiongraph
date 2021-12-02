from abc import abstractmethod, ABC

import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class FaceLandmarkEstimator(LandmarkEstimator, RoiEstimator, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> FaceLandmarkResult:
        pass
