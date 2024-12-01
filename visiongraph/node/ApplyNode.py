from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

InputType = TypeVar('InputType')


class ApplyNode(GraphNode[InputType, ResultDict]):
    """
    A graph node that applies transformations to input data and returns the results.
    """

    def __init__(self, **nodes: GraphNode):
        """
        Initializes an ApplyNode with a dictionary of child nodes.

        :param nodes: A dictionary mapping node names to their corresponding GraphNode instances.
        """
        self.nodes = nodes

    def setup(self):
        """
        Sets up the child nodes by calling their setup methods.
        """
        for node in self.nodes.values():
            node.setup()

    def process(self, data: InputType) -> ResultDict:
        """
        Applies the transformations defined in this node to the input data and returns the results.

        :param data: The input data to be transformed.

        :return: A dictionary containing the transformation results.
        """
        results = ResultDict()
        for name, node in self.nodes.items():
            results[name] = node.process(data)
        return results

    def release(self):
        """
        Releases any resources held by this node and its child nodes.
        """
        for node in self.nodes.values():
            node.release()

    def configure(self, args: Namespace):
        """
        Configures the transformation parameters based on the provided command-line arguments.

        :param args: The parsed command-line arguments.
        """
        for node in self.nodes.values():
            node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser) -> None:
        """
        Adds command-line parameters to the parser that can be used to configure this node's behavior.

        :param parser: The parser instance to which parameters are being added.
        """
        pass
