import logging
from argparse import ArgumentParser, Namespace
from typing import TypeVar, Optional, Union

import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict

OutputType = TypeVar("OutputType")


class BreakpointNode(GraphNode[Union[np.ndarray, ResultDict], Optional[Union[np.ndarray, ResultDict]]]):
    def __init__(self):
        pass

    def setup(self):
        pass

    def process(self, data: Union[np.ndarray, ResultDict]) -> Optional[Union[np.ndarray, ResultDict]]:
        breakpoint()
        return data

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
