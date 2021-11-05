from abc import ABC, abstractmethod
from typing import List

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult


class PoseEstimator(LandmarkEstimator, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> List[PoseLandmarkResult]:
        pass
