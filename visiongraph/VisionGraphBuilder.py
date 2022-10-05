from typing import Callable, Optional

from visiongraph.GraphNode import GraphNode
from visiongraph.VisionGraph import VisionGraph
from visiongraph.input.BaseInput import BaseInput
from visiongraph.node.ApplyNode import ApplyNode
from visiongraph.node.CustomNode import CustomNode
from visiongraph.node.ExtractNode import ExtractNode
from visiongraph.node.PassThroughNode import PassThroughNode
from visiongraph.node.SequenceNode import SequenceNode


def sequence(*nodes: GraphNode) -> SequenceNode:
    return SequenceNode(*nodes)


def passthrough() -> PassThroughNode:
    return PassThroughNode()


def custom(method: Callable, *args, **kwargs) -> CustomNode:
    return CustomNode(method, *args, **kwargs)


def extract(key: str, drop: bool = False) -> ExtractNode:
    return ExtractNode(key, drop)


class _VisionGraphBuilder:
    def __init__(self, graph: VisionGraph):
        self._graph = graph

    def then(self, *nodes: GraphNode) -> "_VisionGraphBuilder":
        for node in nodes:
            self._graph.add_nodes(node)
        return self

    def apply(self, **nodes: GraphNode) -> "_VisionGraphBuilder":
        self._graph.add_nodes(ApplyNode(**nodes))
        return self

    def build(self) -> VisionGraph:
        return self._graph

    def open(self) -> VisionGraph:
        graph = self.build()
        graph.open()
        return graph


def create_graph(input_node: Optional[BaseInput] = None, name: str = "VisionGraph", multi_threaded: bool = False,
                 daemon: bool = False, handle_signals: bool = False) -> _VisionGraphBuilder:
    return _VisionGraphBuilder(VisionGraph(input=input_node, name=name, multi_threaded=multi_threaded,
                                           daemon=daemon, handle_signals=handle_signals))
