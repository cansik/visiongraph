from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.BaseResult import BaseResult

OutputType = TypeVar('OutputType', bound=BaseResult)


class VisionEstimator(BaseEstimator[np.ndarray, OutputType], ABC):
    """
    Abstract base class for estimators in the VisionGraph framework.

    This class provides a common interface for different types of estimators,
    allowing them to be used uniformly throughout the framework.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> OutputType:
        """
        Processes the input data using the estimator's logic.

        :param data: The input data to be processed.

        :return: The result of processing the input data.
        """
        pass
