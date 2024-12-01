from abc import abstractmethod, ABC
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.HandLandmarkResult import HandLandmarkResult

OutputType = TypeVar('OutputType', bound=HandLandmarkResult)


class HandLandmarkEstimator(LandmarkEstimator[OutputType], RoiEstimator, ABC):
    """
    Abstract base class for hand landmark estimators.

    This class provides a common interface for various hand landmark estimation algorithms,
    inheriting from both LandmarkEstimator and RoiEstimator to leverage their capabilities.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Processes the input data and returns a list of results.

        :param data: The input data to be processed.

        :return: A list of results, where each result is an instance of OutputType.
        """
        pass
