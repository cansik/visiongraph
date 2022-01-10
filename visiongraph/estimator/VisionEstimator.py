from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.BaseResult import BaseResult

OutputType = TypeVar('OutputType', bound=BaseResult)


class VisionEstimator(BaseEstimator[np.ndarray, OutputType], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> OutputType:
        pass
