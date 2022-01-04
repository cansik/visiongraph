from abc import ABC, abstractmethod
from typing import Any

from visiongraph.model.chain.ChainableNode import ChainableNode
from visiongraph.result.BaseResult import BaseResult


class BaseEstimator(ChainableNode, ABC):
    @abstractmethod
    def estimate(self, *args, **kwargs) -> BaseResult:
        pass

    def _chain_apply(self, *args, **kwargs) -> Any:
        return self.estimate(*args, **kwargs)
