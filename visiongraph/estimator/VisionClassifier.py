from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace
from typing import TypeVar

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.ClassificationResult import ClassificationResult

OutputType = TypeVar('OutputType', bound=ClassificationResult)


class VisionClassifier(VisionEstimator[OutputType], BaseClassifier[np.ndarray, OutputType], ABC):
    """
    Abstract base class for vision classifiers.

    Provides a common interface for various vision classification algorithms.
    """

    @abstractmethod
    def process(self, data: np.ndarray) -> OutputType:
        """
        Processes the input data and returns a classification result.

        :param data: Input data to be processed.

        :return: Classification result.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the vision classifier with command-line arguments.

        Calls the `configure` methods of the parent classes (`VisionEstimator` and `BaseClassifier`) and then proceeds with its own configuration.
        """
        VisionEstimator.configure(self, args)
        BaseClassifier.configure(self, args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters for the vision classifier to the parser.

        Calls the `add_params` methods of the parent classes (`VisionEstimator` and `BaseClassifier`) and then adds its own parameters.
        """
        VisionEstimator.add_params(parser)
        BaseClassifier.add_params(parser)
