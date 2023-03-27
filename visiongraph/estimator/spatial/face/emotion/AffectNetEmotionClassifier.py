from argparse import ArgumentParser, Namespace

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.face.emotion.FaceEmotionEstimator import FaceEmotionEstimator
from visiongraph.model.types.ModelPrecision import ModelPrecision
from visiongraph.result.spatial.face.EmotionClassificationResult import EmotionClassificationResult


class AffectNetEmotionClassifier(FaceEmotionEstimator):
    def __init__(self, model_precision: ModelPrecision = ModelPrecision.FP32, device: str = "AUTO"):
        super().__init__(min_score=0.5)

        model_name = f"emotions-recognition-retail-0003-{model_precision.open_vino_model_suffix}"
        model, weights = RepositoryAsset.openVino(model_name)
        self.engine = VisionInferenceEngine(model, weights, device=device)

        self.labels = ["neutral", "happy", "sad", "surprise", "anger"]

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> EmotionClassificationResult:
        output = self.engine.process(data)
        probability = np.squeeze(output["prob_emotion"])
        best_index = int(np.argmax(probability))

        return EmotionClassificationResult(best_index, self.labels[best_index],
                                           float(probability[best_index]), probability)

    def _transform_result(self, result: EmotionClassificationResult, image: np.ndarray,
                          roi: np.ndarray, xs: float, ys: float):
        pass

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
