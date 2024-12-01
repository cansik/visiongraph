from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import TypeVar

import numpy as np

from visiongraph import GraphNode
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.BaseResult import BaseResult

OutputType = TypeVar('OutputType', bound=BaseResult)


class ChainEstimator(VisionEstimator[OutputType], ABC):
    """
    A base class for chain estimators in the VisionGraph framework.

    Provides a common interface for chaining multiple estimators together,
    allowing for modular and extensible vision graph processing pipelines.
    """

    def __init__(self, *links: GraphNode):
        """
        Initializes the ChainEstimator with a list of linked nodes.

        :param links: The nodes to be linked in the chain.
        """
        self.links = links

    def setup(self):
        """
        Sets up the estimator by calling the setup method on each linked node.

        This ensures that all nodes in the chain are properly configured before processing begins.
        """
        super().setup()
        for link in self.links:
            link.setup()

    def process(self, image: np.ndarray) -> OutputType:
        """
        Processes an input image through the chain of linked nodes.

        :param image: The input image to be processed.

        :return: The output result from the final node in the chain.
        """
        current_data = image
        for link in self.links:
            current_data = link.process(current_data)
        return current_data

    def release(self):
        """
        Releases any resources held by the estimator and its linked nodes.

        This is typically called at the end of processing to ensure proper cleanup.
        """
        super().release()
        for link in self.links:
            link.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator with command-line arguments.

        :param args: The parsed command-line arguments.
        """
        super().configure(args)
        for link in self.links:
            link.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the given ArgumentParser instance for the ChainEstimator.

        This is typically called from a setup function or class method to define command-line arguments.

        :param parser: The parser instance to add parameters to.
        """
        super().add_params(parser)
