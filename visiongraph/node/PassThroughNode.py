from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class PassThroughNode(GraphNode[InputType, OutputType]):
    def setup(self):
        pass

    def process(self, data: InputType) -> OutputType:
        return data

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
