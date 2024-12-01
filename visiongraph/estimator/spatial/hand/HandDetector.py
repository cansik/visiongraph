from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.HandDetectionResult import HandDetectionResult

OutputType = TypeVar('OutputType', bound=HandDetectionResult)


class HandDetector(ObjectDetector[OutputType], ABC):
    """
    Abstract base class for hand detectors.

    This class provides a common interface for different hand detection algorithms,
    allowing users to easily switch between them based on their specific requirements.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Processes the input data using the detector's algorithm.

        :param data: The input data to be processed.

        :return: A list of detection results.
        """
        pass
