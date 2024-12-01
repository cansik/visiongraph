from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

OutputType = TypeVar('OutputType', bound=LandmarkDetectionResult)


class LandmarkEstimator(ObjectDetector[OutputType], ABC):
    """
    A generic abstract class for a landmark estimator that extends ObjectDetector.

    This class is responsible for processing input data and generating a list of LandmarkDetectionResults.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        An abstract method to process input data and return a list of output type results.

        :param data: An input NumPy array representing the data to be processed.

        :return: A list of output type results.
        """
        pass
