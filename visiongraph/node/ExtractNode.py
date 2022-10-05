import logging
from argparse import ArgumentParser, Namespace
from typing import TypeVar, Optional

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

OutputType = TypeVar("OutputType")


class ExtractNode(GraphNode[ResultDict, Optional[OutputType]]):
    def __init__(self, key: str, drop: bool = False):
        self.key = key
        self.drop = drop

    def setup(self):
        pass

    def process(self, data: ResultDict) -> Optional[OutputType]:
        if self.key not in data:
            logging.error(f"Could not find key {self.key} in result-dict {data}.")
            return None

        if self.drop:
            return data.pop(self.key)

        return data[self.key]

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
