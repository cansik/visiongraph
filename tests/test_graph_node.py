import unittest
from argparse import ArgumentParser, Namespace

from visiongraph import vg
from visiongraph.GraphNode import InputType, OutputType


class DemoNode(vg.GraphNode[int, int]):
    def __init__(self):
        self.counter = 0

    def setup(self) -> None:
        self.counter += 1

    def process(self, data: InputType) -> OutputType:
        pass

    def release(self) -> None:
        self.counter += 10

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


class GraphNodeTests(unittest.TestCase):
    def test_context_manager_pattern(self):
        with DemoNode() as node:
            node.process(1)

        self.assertEqual(node.counter, 11, "Context manager seems not to work.")


if __name__ == '__main__':
    unittest.main()
