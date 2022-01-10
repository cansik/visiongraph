from abc import abstractmethod, ABC
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult

OutputType = TypeVar('OutputType', bound=FaceLandmarkResult)


class FaceLandmarkEstimator(LandmarkEstimator[OutputType], RoiEstimator, ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> ResultList[OutputType]:
        pass
