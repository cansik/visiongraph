from abc import abstractmethod, ABC
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.HandLandmarkResult import HandLandmarkResult

OutputType = TypeVar('OutputType', bound=HandLandmarkResult)


class HandLandmarkEstimator(LandmarkEstimator[OutputType], RoiEstimator, ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        pass
