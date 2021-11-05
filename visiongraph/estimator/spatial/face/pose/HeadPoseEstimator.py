from abc import abstractmethod

import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult


class HeadPoseEstimator(RoiEstimator):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> HeadPoseResult:
        pass
