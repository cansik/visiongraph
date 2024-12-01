from typing import Callable, Optional

from visiongraph.GraphNode import GraphNode
from visiongraph.VisionGraph import VisionGraph
from visiongraph.input.BaseInput import BaseInput
from visiongraph.node.ApplyNode import ApplyNode
from visiongraph.node.BreakpointNode import BreakpointNode
from visiongraph.node.CustomNode import CustomNode
from visiongraph.node.ExtractNode import ExtractNode
from visiongraph.node.PassThroughNode import PassThroughNode
from visiongraph.node.SequenceNode import SequenceNode


def sequence(*nodes: GraphNode) -> SequenceNode:
    """
    Creates a new Sequence node with the given nodes.

    :param *nodes: The nodes to be added to the Sequence node.
    :return: The newly created Sequence node.
    """
    return SequenceNode(*nodes)


def passthrough() -> PassThroughNode:
    """
    Creates a new PassThrough node.

    :return: The newly created PassThrough node.
    """
    return PassThroughNode()


def custom(method: Callable, *args, **kwargs) -> CustomNode:
    """
    Creates a new Custom node with the given method and arguments.

    :param method: The function to be called in the Custom node.
    :param *args: The positional arguments for the Custom node.
    :param **kwargs: The keyword arguments for the Custom node.
    :return: The newly created Custom node.
    """
    return CustomNode(method, *args, **kwargs)


def extract(key: str, drop: bool = False) -> ExtractNode:
    """
    Creates a new Extract node with the given key and drop flag.

    :param key: The key for which to extract data.
    :param drop: Whether to drop the extracted value. Defaults to False.
    :return: The newly created Extract node.
    """
    return ExtractNode(key, drop)


def add_breakpoint() -> BreakpointNode:
    """
    Creates a new Breakpoint node.

    :return: The newly created Breakpoint node.
    """
    return BreakpointNode()


class _VisionGraphBuilder:
    def __init__(self, graph: VisionGraph):
        """
        Initializes the VisionGraphBuilder with the given VisionGraph instance.

        :param graph: The VisionGraph instance to be used for building.
        """
        self._graph = graph

    def then(self, *nodes: GraphNode) -> "_VisionGraphBuilder":
        """
        Adds the given nodes to the current build and returns the builder.

        :param *nodes: The nodes to be added to the Sequence node.
        :return: The updated builder instance.
        """
        for node in nodes:
            self._graph.add_nodes(node)
        return self

    def apply(self, **nodes: GraphNode) -> "_VisionGraphBuilder":
        """
        Adds an ApplyNode to the current build and returns the builder.

        :param **nodes: The arguments for the ApplyNode.
        :return: The updated builder instance.
        """
        self._graph.add_nodes(ApplyNode(**nodes))
        return self

    def build(self) -> VisionGraph:
        """
        Builds the current graph and returns it.

        :return: The built VisionGraph instance.
        """
        return self._graph

    def open(self) -> VisionGraph:
        """
        Opens the currently built graph and returns it.

        :return: The opened VisionGraph instance.
        """
        graph = self.build()
        graph.open()
        return graph


def create_graph(input_node: Optional[BaseInput] = None, name: str = "VisionGraph", multi_threaded: bool = False,
                 daemon: bool = False, handle_signals: bool = False) -> _VisionGraphBuilder:
    """
    Creates a new VisionGraph instance with the given input node and options.

    :param input_node: The input node for the graph. Defaults to None.
    :param name: The name of the graph. Defaults to "VisionGraph".
    :param multi_threaded: Whether to run the graph in multiple threads. Defaults to False.
    :param daemon: Whether to run the graph as a daemon process. Defaults to False.
    :param handle_signals: Whether to handle signals in the graph. Defaults to False.

    :return: The builder instance for creating the VisionGraph.
    """
    return _VisionGraphBuilder(VisionGraph(input=input_node, name=name, multi_threaded=multi_threaded,
                                           daemon=daemon, handle_signals=handle_signals))
