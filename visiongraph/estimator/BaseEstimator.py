from abc import ABC, abstractmethod
from typing import TypeVar

from visiongraph.GraphNode import GraphNode
from visiongraph.result.BaseResult import BaseResult

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=BaseResult)


class BaseEstimator(GraphNode[InputType, OutputType], ABC):
    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        pass
