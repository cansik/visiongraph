from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.special import softmax

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.face.emotion.FaceEmotionEstimator import FaceEmotionEstimator
from visiongraph.result.spatial.face.EmotionClassificationResult import EmotionClassificationResult


class FERPlusEmotionClassifier(FaceEmotionEstimator):
    """
    A class to represent an example object with a name and a status flag.

    Provides an implementation of the Face Emotion Estimator using FER+ models.
    """

    def __init__(self, model: Asset = RepositoryAsset("emotion-ferplus-8.onnx"),
                 engine: InferenceEngine = InferenceEngine.ONNX):
        """
        Initializes the FERPlusEmotionClassifier object.

        :param model: The ONNX model to be used for emotion classification.
        :param engine: The inference engine type, e.g., ONNX or Tensorflow Lite.
        """
        super().__init__(0.5)

        self.model = model
        self.engine = InferenceEngineFactory.create(engine, [model],
                                                    flip_channels=False,
                                                    scale=1.0,
                                                    transpose=False,
                                                    padding=False)

        self.labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

    def setup(self):
        """
        Sets up the inference engine.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> EmotionClassificationResult:
        """
        Processes an input image and returns an EmotionClassificationResult object.

        :param image: The input image to be processed.

        :return: An object containing the best class index, label,
        """
        output = self.engine.process(image)
        probability = softmax(np.squeeze(output[self.engine.output_names[0]]))
        best_index = int(np.argmax(probability))

        return EmotionClassificationResult(best_index, self.labels[best_index],
                                           float(probability[best_index]), probability)

    def _transform_result(self, result: EmotionClassificationResult, image: np.ndarray,
                          roi: np.ndarray, xs: float, ys: float):
        pass

    def release(self):
        """
        Releases the inference engine resources.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
