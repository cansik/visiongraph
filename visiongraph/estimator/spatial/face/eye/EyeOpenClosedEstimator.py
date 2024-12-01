from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.special import softmax

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult
from visiongraph.result.spatial.face.EyeOpenClosedResult import EyeOpenClosedResult


class EyeOpenClosedEstimator(RoiEstimator, BaseClassifier):
    """
    A class to estimate whether an eye is open or closed.
    """

    def __init__(self, device: str = "AUTO"):
        """
        Initializes the EyeOpenClosedEstimator object.

        :param device: The target device. Defaults to "AUTO".
        """
        super().__init__(0.5)
        model, weights = RepositoryAsset.openVino("open-closed-eye-0001-fp32")
        self.engine = OpenVinoEngine(model, weights, device=device)

        self.labels = ["closed", "open"]

    def setup(self):
        """
        Sets up the engine for processing.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> EyeOpenClosedResult:
        """
        Processes a given image and returns an EyeOpenClosedResult object.

        :param data: The input image.

        :return: The result of the estimation.
        """
        output = self.engine.process(data)

        probability = softmax(np.squeeze(output[self.engine.output_names[0]]))
        best_index = int(np.argmax(probability))

        return EyeOpenClosedResult(best_index, self.labels[best_index],
                                   float(probability[best_index]), probability)

    def _transform_result(self, result: HeadPoseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        """
        Transforms the result of head pose estimation.

        :param result: The head pose result.
        :param image: The input image.
        :param roi: The region of interest.
        :param xs: The x-coordinate of the ROI.
        :param ys: The y-coordinate of the ROI.
        """
        pass

    def release(self):
        """
        Releases the engine resources.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on the provided arguments.

        :param args: The parser namespace containing the command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the argument parser.

        :param parser: The argument parser object.
        """
        pass
