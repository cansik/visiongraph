from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import List

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.model.chain.ChainableNode import ChainableNode
from visiongraph.result.BaseResult import BaseResult


class ChainEstimator(VisionEstimator, ABC):
    def __init__(self, *links: ChainableNode):
        self.links = links

    def setup(self):
        super().setup()
        for link in self.links:
            link.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> BaseResult:
        current_data = image
        for link in self.links:
            current_data = link._chain_apply(current_data)
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
