from abc import ABC, abstractmethod

from visiongraph.PipelineNode import PipelineNode
from visiongraph.result.BaseResult import BaseResult


class BaseEstimator(PipelineNode, ABC):
    @abstractmethod
    def estimate(self, **kwargs) -> BaseResult:
        pass
