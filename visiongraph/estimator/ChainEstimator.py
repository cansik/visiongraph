from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import TypeVar

import numpy as np

from visiongraph import GraphNode
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.BaseResult import BaseResult

OutputType = TypeVar('OutputType', bound=BaseResult)


class ChainEstimator(VisionEstimator[OutputType], ABC):
    def __init__(self, *links: GraphNode):
        self.links = links

    def setup(self):
        super().setup()
        for link in self.links:
            link.setup()

    def process(self, image: np.ndarray) -> OutputType:
        current_data = image
        for link in self.links:
            current_data = link.process(current_data)
        return current_data

    def release(self):
        super().release()
        for link in self.links:
            link.release()

    def configure(self, args: Namespace):
        super().configure(args)
        for link in self.links:
            link.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        super().add_params(parser)
