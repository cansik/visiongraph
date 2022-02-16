from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

OutputType = TypeVar('OutputType', bound=PoseLandmarkResult)


class PoseEstimator(LandmarkEstimator[OutputType], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        pass
