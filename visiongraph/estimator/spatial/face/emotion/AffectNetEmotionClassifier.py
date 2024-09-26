from argparse import ArgumentParser, Namespace

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.face.emotion.FaceEmotionEstimator import FaceEmotionEstimator
from visiongraph.model.types.ModelPrecision import ModelPrecision
from visiongraph.result.spatial.face.EmotionClassificationResult import EmotionClassificationResult


class AffectNetEmotionClassifier(FaceEmotionEstimator):
    """
    A class to represent an AffectNet Emotion Classifier, inheriting from FaceEmotionEstimator.
    """

    def __init__(self, model_precision: ModelPrecision = ModelPrecision.FP32, device: str = "AUTO"):
        """
        Initializes an instance of AffectNetEmotionClassifier.

        Args:
            model_precision (ModelPrecision): The precision of the model used for classification. Defaults to ModelPrecision.FP32.
            device (str): The device on which the model is executed. Defaults to "AUTO".
        """
        super().__init__(min_score=0.5)

        model_name = f"emotions-recognition-retail-0003-{model_precision.open_vino_model_suffix}"
        model, weights = RepositoryAsset.openVino(model_name)
        self.engine = OpenVinoEngine(model, weights, device=device)

        self.labels = ["neutral", "happy", "sad", "surprise", "anger"]

    def setup(self):
        """
        Sets up the OpenVino engine for processing.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> EmotionClassificationResult:
        """
        Processes input data and returns an EmotionClassificationResult.

        Args:
            data (np.ndarray): The input data to be processed.

        Returns:
            EmotionClassificationResult: An object containing the best emotion label and its corresponding probability.
        """
        output = self.engine.process(data)
        probability = np.squeeze(output["prob_emotion"])
        best_index = int(np.argmax(probability))

        return EmotionClassificationResult(best_index, self.labels[best_index],
                                           float(probability[best_index]), probability)

    def _transform_result(self, result: EmotionClassificationResult, image: np.ndarray,
                          roi: np.ndarray, xs: float, ys: float):
        """
        Transforms the result of an emotion classification.

        Args:
            result (EmotionClassificationResult): The result to be transformed.
            image (np.ndarray): The original input image.
            roi (np.ndarray): The region of interest (ROI) from the image.
            xs (float): The x-coordinate of the ROI.
            ys (float): The y-coordinate of the ROI.
        """
        pass

    def release(self):
        """
        Releases any resources held by the engine.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator with the provided command-line arguments.

        Args:
            args (Namespace): The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for configuration and parsing of command-line arguments.

        Args:
            parser (ArgumentParser): The parser instance to be updated with parameters.
        """
        pass
