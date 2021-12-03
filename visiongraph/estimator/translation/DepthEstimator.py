from abc import ABC, abstractmethod

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.DepthMap import DepthMap


class DepthEstimator(VisionEstimator, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> DepthMap:
        pass
