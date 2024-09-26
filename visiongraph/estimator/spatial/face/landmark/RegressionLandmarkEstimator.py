from argparse import ArgumentParser, Namespace

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.VisionClassifier import VisionClassifier
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.spatial.face.RegressionFace import RegressionFace
from visiongraph.util.VectorUtils import list_of_vector4D


class RegressionLandmarkEstimator(VisionClassifier[RegressionFace], RoiEstimator):
    """
    A class that combines the regression face estimator and ROI estimator to estimate landmarks on faces.
    """

    def __init__(self, min_score: float = 0.0, device: str = "AUTO"):
        """
        Initializes the RegressionLandmarkEstimator with the given minimum score and device.

        Args:
            min_score (float, optional): The minimum confidence score for face detection. Defaults to 0.0.
            device (str, optional): The device to use for inference. Defaults to "AUTO".
        """
        super().__init__(min_score)
        model, weights = RepositoryAsset.openVino("landmarks-regression-retail-0009")
        self.engine = OpenVinoEngine(model, weights, device=device)

    def setup(self):
        """
        Sets up the engine for inference.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> RegressionFace:
        """
        Processes the input data through the engine.

        Args:
            data (np.ndarray): The input data to be processed.

        Returns:
            RegressionFace: The estimated regression face with landmarks.
        """
        outputs = self.engine.process(data)
        output = outputs[self.engine.output_names[0]].reshape((-1, 2))
        result = []

        for point in output:
            result.append((float(point[0]), float(point[1]), 0.0, 1.0))

        return RegressionFace(1.0, list_of_vector4D(result))

    def _transform_result(self, result: RegressionFace, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        """
        Transforms the estimated landmarks based on the ROI and image.

        Args:
            result (RegressionFace): The estimated regression face with landmarks.
            image (np.ndarray): The input image.
            roi (np.ndarray): The region of interest.
            xs (float): The x-coordinate offset.
            ys (float): The y-coordinate offset.
        """
        hi, wi = image.shape[:2]
        hr, wr = roi.shape[:2]

        for i, lm in enumerate(result.landmarks):
            x = ((lm.x * wr) + xs) / float(wi)
            y = ((lm.y * hr) + ys) / float(hi)
            result.landmarks.x[i] = x
            result.landmarks.y[i] = y

    def release(self):
        """
        Releases the engine resources.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on the given arguments.

        Args:
            args (Namespace): The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for configuring the estimator.

        Args:
            parser (ArgumentParser): The parser instance.
        """
        pass
