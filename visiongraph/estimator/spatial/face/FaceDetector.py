from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceDetectionResult import FaceDetectionResult

OutputType = TypeVar('OutputType', bound=FaceDetectionResult)


class FaceDetector(ObjectDetector[OutputType], ABC):
    """
    Abstract base class for face detectors.

    This class provides the basic structure and interface for face detector objects.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> ResultList[OutputType]:
        """
        Processes an input image using the face detection algorithm.

        Args:
            image (np.ndarray): The input image to be processed.

        Returns:
            ResultList[OutputType]: A list of face detection results.
        """

        pass
