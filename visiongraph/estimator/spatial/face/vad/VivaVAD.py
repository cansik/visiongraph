from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Sequence, List

import numpy as np
from scipy.special import softmax

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.face.VivaVADResult import VivaVADResult, NON_SPEAKING_LABEL, SPEAKING_LABEL


class VivaVADConfig(Enum):
    """
    Configuration options for the VivaVAD model.
    """
    I_TCN_148_10_2 = (RepositoryAsset("viva-10-2-148-simplified.onnx"), BlazeFaceMesh.FEATURES_148, 10)
    I_TCN_148_15_2 = (RepositoryAsset("viva-15-2-148-simplified.onnx"), BlazeFaceMesh.FEATURES_148, 15)
    I_TCN_148_30_1 = (RepositoryAsset("viva-30-1-148-simplified.onnx"), BlazeFaceMesh.FEATURES_148, 30)


class VivaVAD(BaseClassifier[List[np.ndarray], ResultList[VivaVADResult]]):
    """
    A voice activity detection (VAD) classifier that processes sequences of facial landmarks
    to determine speaking activity.
    """

    def __init__(self, *assets: Asset,
                 landmark_indices: Sequence[int] = BlazeFaceMesh.FEATURES_148,
                 sequence_length: int = 15,
                 min_score: float = 0.75,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        """
        Initializes the VivaVAD classifier.

        :param assets: Model assets required for inference.
        :param landmark_indices: Indices of facial landmarks to process.
        :param sequence_length: The length of the input sequence for inference.
        :param min_score: Minimum score threshold for classification results.
        :param engine: The inference engine used for model execution.
        """
        super().__init__(min_score)

        self.engine = InferenceEngineFactory.create(engine, assets, flip_channels=False, transpose=False)
        self.sequence_length = sequence_length
        self.landmark_indices = landmark_indices

        self.labels = [NON_SPEAKING_LABEL, SPEAKING_LABEL]

    def setup(self) -> None:
        """
        Sets up the inference engine for processing.
        """
        self.engine.setup()

    def process(self, samples: List[np.ndarray]) -> ResultList[VivaVADResult]:
        """
        Processes a sequence of facial landmarks to classify speaking activity.

        :param samples: A list of numpy arrays representing sequences of facial landmarks.

        :return: A result list containing VivaVAD classification results.
        """
        landmarks_per_face = np.array(samples)

        # Set dynamic input shape for the inference engine
        self.engine.dynamic_input_shapes[self.engine.first_input_name] = list(landmarks_per_face.shape)
        inputs = {self.engine.first_input_name: landmarks_per_face}

        # Run inference
        outputs = self.engine.predict(inputs)

        # Interpret the results
        results = ResultList()
        for logits in outputs[self.engine.output_names[0]]:
            scores = softmax(logits)
            predicted_class = np.argmax(logits)

            results.append(VivaVADResult(predicted_class, self.labels[predicted_class],
                                         float(scores[predicted_class]), logits))

        return results

    def release(self) -> None:
        """
        Releases resources held by the inference engine.
        """
        self.engine.release()

    @staticmethod
    def create(config: VivaVADConfig = VivaVADConfig.I_TCN_148_10_2,
               engine: InferenceEngine = InferenceEngine.ONNX) -> "VivaVAD":
        """
        Creates a VivaVAD classifier with the specified configuration and engine.

        :param config: The VivaVAD model configuration to use.
        :param engine: The inference engine to use for model execution.

        :return: An instance of the VivaVAD classifier.
        """
        model, landmark_indices, sequence_length = config.value
        return VivaVAD(model, landmark_indices=landmark_indices,
                       sequence_length=sequence_length, engine=engine)

    def configure(self, args: Namespace):
        """
        Configures the VivaVAD classifier using command-line arguments.

        :param args: Namespace containing configuration arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters specific to the VivaVAD classifier.

        :param parser: The argument parser to configure.
        """
        pass
