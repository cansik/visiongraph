from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult

OutputType = TypeVar('OutputType', bound=InstanceSegmentationResult)


class InstanceSegmentationEstimator(ObjectDetector[OutputType], ABC):
    """
    Abstract base class for instance segmentation estimators that detect objects in spatial images.

    This class extends ObjectDetector and provides an abstract method for processing data and returning a ResultList of OutputType.

    Type Parameters:
        - OutputType: Type of the output result, bound to InstanceSegmentationResult.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Abstract method to process input data and return a list of instance segmentation results of type OutputType.

        :param data: Input data in the form of a numpy array.

        :return: A list of instance segmentation results of type OutputType.
        """
        pass
