import logging
from argparse import ArgumentParser, Namespace
from typing import TypeVar, Optional

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

OutputType = TypeVar("OutputType")


class ExtractNode(GraphNode[ResultDict, Optional[OutputType]]):
    """
    A node in the graph that extracts a key-value pair from a ResultDict.
    """

    def __init__(self, key: str, drop: bool = False):
        """
        Initializes an ExtractNode with a specific key and optional drop behavior.

        :param key: The key to be extracted.
        :param drop: If True, removes the key-value pair after extraction. Defaults to False.
        """
        self.key = key
        self.drop = drop

    def setup(self):
        """
        Sets up the node by initializing its internal state.
        """
        pass

    def process(self, data: ResultDict) -> Optional[OutputType]:
        """
        Extracts the value associated with the specified key from the input data.

        If the key is not found or drop is True, returns None or removes the key-value pair.

        :param data: The input data containing the key-value pair to be extracted.

        :return: The extracted value if it exists and drop is False, otherwise None.
        """
        if self.key not in data:
            logging.error(f"Could not find key {self.key} in result-dict {data}.")
            return None

        if self.drop:
            return data.pop(self.key)

        return data[self.key]

    def release(self):
        """
        Releases any resources held by the node.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the node with command-line arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters for the ExtractNode to the specified parser.

        :param parser: The parser to which the node's parameters will be added.
        """
        pass
