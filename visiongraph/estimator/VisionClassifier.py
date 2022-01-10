from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace
from typing import TypeVar

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.ClassificationResult import ClassificationResult

OutputType = TypeVar('OutputType', bound=ClassificationResult)


class VisionClassifier(VisionEstimator[OutputType], BaseClassifier[np.ndarray, OutputType], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> OutputType:
        pass

    def configure(self, args: Namespace):
        VisionEstimator.configure(self, args)
        BaseClassifier.configure(self, args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        VisionEstimator.add_params(parser)
        BaseClassifier.add_params(parser)
