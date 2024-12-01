from abc import abstractmethod, ABC
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult

OutputType = TypeVar('OutputType', bound=FaceLandmarkResult)


class FaceLandmarkEstimator(LandmarkEstimator[OutputType], RoiEstimator, ABC):
    """
    An abstract class to estimate face landmarks from images.

    This class inherits from both LandmarkEstimator and RoiEstimator,
    which are not shown in this snippet. It is intended to be subclassed
    for specific face landmark estimation tasks.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> ResultList[OutputType]:
        """
        Processes the input image to extract face landmarks.

        :param image: The input image to be processed.

        :return: A list of face landmark results.
        """
        pass
