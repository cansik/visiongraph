from argparse import ArgumentParser, Namespace
from typing import TypeVar, Callable

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class CustomNode(GraphNode[InputType, OutputType]):
    """
    A custom node for graph-based computations.
    """

    def __init__(self, method: Callable[[InputType], OutputType], *args, **kwargs):
        """
        Initializes a new instance of CustomNode.

        :param method: The function to be executed on the input data.
        :param *args: Additional positional arguments passed to the `method`.
        :param **kwargs: Additional keyword arguments passed to the `method`.

        """
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def setup(self):
        """
        Setup the node for execution.
        """
        pass

    def process(self, data: InputType) -> OutputType:
        """
        Executes the `method` on the input data.

        :param data: The input data to be processed.

        :return: The result of executing the `method`.
        """
        result = self.method(data, *self.args, **self.kwargs)

        if result is None:
            return data

        return result

    def release(self):
        """
        Releases any resources held by the node.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the node based on the provided arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for this custom node.

        :param parser: The parser to add parameters to.
        """
        pass
