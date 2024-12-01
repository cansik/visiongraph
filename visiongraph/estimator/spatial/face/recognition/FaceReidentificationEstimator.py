from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.face.recognition.FaceRecognitionEstimator import FaceRecognitionEstimator
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class FaceReidentificationConfig(Enum):
    """
    Enumerates possible face re-identification configurations.
    """

    Retail_0095_FP16_INT8 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp16-int8")
    Retail_0095_FP16 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp16")
    Retail_0095_FP32 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp32")


class FaceReidentificationEstimator(FaceRecognitionEstimator):
    """
    Estimator for face re-identification tasks.
    """

    def __init__(self, model: Asset, weights: Asset, device: str = "AUTO"):
        """
        Initializes the estimator with a given model and weights.

        :param model: The OpenVINO model to use.
        :param weights: The OpenVINO weights to use.
        :param device: The target device for inference. Defaults to "AUTO".
        """
        super().__init__()
        self.engine = OpenVinoEngine(model, weights, flip_channels=True, device=device)

        # left eye, right eye, tip of nose, left lip corner, right lip corner
        # https://docs.openvino.ai/latest/omz_models_model_face_reidentification_retail_0095.html
        self.normalized_keypoints = np.array([[0.31556875000000000, 0.4615741071428571],
                                              [0.68262291666666670, 0.4615741071428571],
                                              [0.50026249999999990, 0.6405053571428571],
                                              [0.34947187500000004, 0.8246919642857142],
                                              [0.65343645833333330, 0.8246919642857142]
                                              ], dtype=np.float32)

    def setup(self):
        """
        Sets up the estimator for inference.
        """
        self.engine.setup()

    def process(self, image: np.ndarray, landmarks: Optional[FaceLandmarkResult] = None) -> EmbeddingResult:
        """
        Processes a given image and optional face landmarks.

        :param image: The input image.
        :param landmarks: Face landmarks. Defaults to None.

        :return: The extracted face embedding.
        """
        image, landmarks = self._pre_process_input(image, landmarks)
        aligned_face, landmark_overlap = self._align_face(image, landmarks, self.normalized_keypoints)

        result = self.engine.process(aligned_face)
        data = result[self.engine.output_names[0]]
        flat_data = data.reshape((data.shape[1]))

        return EmbeddingResult(flat_data)

    def release(self):
        """
        Releases the estimator's resources.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on given arguments.

        :param args: The input arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters for the estimator to an argument parser.
        """
        pass

    @staticmethod
    def create(config: FaceReidentificationConfig = FaceReidentificationConfig.Retail_0095_FP32) -> \
            "FaceReidentificationEstimator":
        """
        Creates a new instance of the estimator based on a given configuration.

        :param config: The configuration to use. Defaults to FaceReidentificationConfig.Retail_0095_FP32.
        """
        model, weights = config.value
        return FaceReidentificationEstimator(model, weights)
