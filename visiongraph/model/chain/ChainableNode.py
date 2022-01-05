from abc import abstractmethod, ABC
from typing import Any

from visiongraph.GraphNode import GraphNode


class ChainableNode(GraphNode, ABC):

    @abstractmethod
    def _chain_apply(self, *args, **kwargs) -> Any:
        pass
