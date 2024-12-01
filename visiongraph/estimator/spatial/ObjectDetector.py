from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.VisionClassifier import VisionClassifier
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

OutputType = TypeVar('OutputType', bound=ObjectDetectionResult)


class ObjectDetector(VisionClassifier[ResultList[OutputType]], ABC):
    """
    Abstract base class for object detectors based on vision classifiers.

    This class inherits from VisionClassifier and is meant to be subclassed for specific object detection tasks.

    Type Parameters:
        OutputType: The type of output expected from the object detection process.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Abstract method to process input data and perform object detection.

        :param data: The input data for object detection, typically an image or a frame.

        :return: A list of object detection results of type OutputType.
        """
        pass
