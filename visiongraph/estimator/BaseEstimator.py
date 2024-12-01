from abc import ABC, abstractmethod
from typing import TypeVar

from visiongraph.GraphNode import GraphNode
from visiongraph.result.BaseResult import BaseResult

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=BaseResult)


class BaseEstimator(GraphNode[InputType, OutputType], ABC):
    """
    Abstract base class for estimators in VisionGraph.

    This class provides a common interface for different types of estimators
    to extend and provide their own implementation of the `process` method.
    """

    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        """
        Processes the input data using the estimator's logic.

        :param data: The input data to be processed.

        :return: The processed output of type `OutputType`.
        """
        pass
