from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class PassThroughNode(GraphNode[InputType, OutputType]):
    """
    A node that passes through input data without modification.

    This class represents a basic building block for graph-based neural networks.
    It maintains a forward pass, release, and configuration process as required by the GraphNode interface.
    """

    def setup(self):
        """
        Initializes the PassThroughNode. No additional setup is required as this node passes through data without modification.

        Returns:
            None
        """
        pass

    def process(self, data: InputType) -> OutputType:
        """
        The forward pass operation of the PassThroughNode, where input data is returned as output.

        Args:
            data (InputType): The input data to be passed through.

        Returns:
            OutputType: The same type of data that was inputted.
        """
        return data

    def release(self):
        """
        Releases any resources held by the PassThroughNode. Since this node does not hold any resources, no action is required here.

        Returns:
            None
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the PassThroughNode based on the provided arguments. This method does nothing as the PassThroughNode's behavior remains constant regardless of configuration.

        Args:
            args (Namespace): The namespace containing command-line arguments.

        Returns:
            None
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser. This method is used by the GraphNode.add_params method and only adds an empty parameter to this specific node's parser.

        Args:
            parser (ArgumentParser): The parser to be extended with additional parameters.

        Returns:
            None
        """
        pass
