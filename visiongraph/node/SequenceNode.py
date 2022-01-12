from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class SequenceNode(GraphNode[InputType, OutputType]):
    def __init__(self, *nodes: GraphNode):
        self.nodes = nodes

    def setup(self):
        for node in self.nodes:
            node.setup()

    def process(self, data: InputType) -> OutputType:
        temp = data
        for node in self.nodes:
            temp = node.process(temp)
        return temp

    def release(self):
        for node in self.nodes:
            node.release()

    def configure(self, args: Namespace):
        for node in self.nodes:
            node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
