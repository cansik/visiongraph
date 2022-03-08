from argparse import ArgumentParser, Namespace
from typing import TypeVar, Callable

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class CustomNode(GraphNode[InputType, OutputType]):
    def __init__(self, method: Callable[[InputType], OutputType]):
        self.method = method

    def setup(self):
        pass

    def process(self, data: InputType) -> OutputType:
        result = self.method(data)

        if result is None:
            return data

        return result

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
