from argparse import ArgumentParser, Namespace
from typing import TypeVar, Optional, Union

import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

OutputType = TypeVar("OutputType")


class BreakpointNode(GraphNode[Union[np.ndarray, ResultDict], Optional[Union[np.ndarray, ResultDict]]]):
    """
    A node in the graph that triggers a breakpoint.
    """

    def __init__(self):
        """
        Initializes the BreakpointNode object.
        """
        pass

    def setup(self):
        """
        Sets up the node for execution. This method should be overridden by subclasses.

        Note:
            The purpose of this method is to perform any necessary initialization before processing data.
        """
        pass

    def process(self, data: Union[np.ndarray, ResultDict]) -> Optional[Union[np.ndarray, ResultDict]]:
        """
        Processes the input data and returns an optional result.

        :param data: The data to be processed.

        :return: The processed result.
        """
        breakpoint()
        return data

    def release(self):
        """
        Releases any resources held by the node. This method should be overridden by subclasses.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the node based on the provided arguments.

        :param args: The namespace containing the command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser) -> None:
        """
        Adds parameters to the parser for this node type.

        :param parser: The parser to be updated.
        """
        pass
