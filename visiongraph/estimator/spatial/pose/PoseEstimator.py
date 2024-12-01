from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult

OutputType = TypeVar('OutputType', bound=PoseLandmarkResult)


class PoseEstimator(LandmarkEstimator[OutputType], ABC):
    """
    Abstract base class for pose estimators.

    Provides a common interface for different pose estimation algorithms.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Processes the input data to estimate poses.

        :param data: Input data containing spatial information.

        :return: A list of pose landmark results.
        """
        pass
