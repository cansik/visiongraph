from abc import abstractmethod, ABC
from typing import Any

from visiongraph.PipelineNode import PipelineNode


class ChainableNode(PipelineNode, ABC):

    @abstractmethod
    def _chain_apply(self, *args, **kwargs) -> Any:
        pass
