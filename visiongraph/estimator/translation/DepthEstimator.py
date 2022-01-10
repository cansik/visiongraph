from abc import ABC, abstractmethod

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.DepthMap import DepthMap


class DepthEstimator(VisionEstimator[DepthMap], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> DepthMap:
        pass
