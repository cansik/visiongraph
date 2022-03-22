from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.ImageResult import ImageResult

InpaintInputType = Dict[str, np.ndarray]


class BaseInpainter(BaseEstimator[InpaintInputType, ImageResult], ABC):

    def process(self, data: InpaintInputType) -> ImageResult:
        return self.inpaint(**data)

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> ImageResult:
        pass
