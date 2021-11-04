from abc import ABC, abstractmethod

import numpy as np

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.BaseResult import BaseResult


class VisionEstimator(BaseEstimator, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> BaseResult:
        pass
