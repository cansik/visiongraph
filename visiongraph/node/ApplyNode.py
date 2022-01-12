from argparse import ArgumentParser, Namespace
from typing import TypeVar

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

InputType = TypeVar('InputType')


class ApplyNode(GraphNode[InputType, ResultDict]):
    def __init__(self, **nodes: GraphNode):
        self.nodes = nodes

    def setup(self):
        for node in self.nodes.values():
            node.setup()

    def process(self, data: InputType) -> ResultDict:
        results = ResultDict()
        for name, node in self.nodes.items():
            results[name] = node.process(data)
        return results

    def release(self):
        for node in self.nodes.values():
            node.release()

    def configure(self, args: Namespace):
        for node in self.nodes.values():
            node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
