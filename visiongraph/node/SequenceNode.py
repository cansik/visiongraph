from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class SequenceNode(GraphNode[InputType, OutputType]):
    """
    Represents a node in a sequence of nodes.
    """

    def __init__(self, *nodes: GraphNode):
        """
        Initializes the SequenceNode with a list of child nodes.

        :param *nodes: The child nodes to be initialized.
        """
        self.nodes = nodes

    def setup(self) -> None:
        """
        Sets up all the child nodes in the sequence.
        """
        for node in self.nodes:
            node.setup()

    def process(self, data: InputType) -> OutputType:
        """
        Processes the input data through each node in the sequence.

        :param data: The input data to be processed.

        :return: The output data after processing.
        """
        temp = data
        for node in self.nodes:
            temp = node.process(temp)
        return temp

    def release(self) -> None:
        """
        Releases the resources held by all the child nodes in the sequence.
        """
        for node in self.nodes:
            node.release()

    def configure(self, args: Namespace) -> None:
        """
        Configures all the child nodes in the sequence based on the provided arguments.

        :param args: The namespace containing the configuration arguments.
        """
        for node in self.nodes:
            node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser) -> None:
        """
        Adds parameters to the parser that are specific to this sequence node.

        :param parser: The parser to be updated.
        """
        pass
