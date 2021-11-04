from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.ClassificationResult import ClassificationResult


class VisionClassifier(VisionEstimator, BaseClassifier, ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, **kwargs) -> ClassificationResult:
        pass

    def configure(self, args: Namespace):
        VisionEstimator.configure(self, args)
        BaseClassifier.configure(self, args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        VisionEstimator.add_params(parser)
        BaseClassifier.add_params(parser)


