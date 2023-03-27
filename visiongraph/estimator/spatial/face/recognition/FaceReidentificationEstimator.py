from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.face.recognition.FaceRecognitionEstimator import FaceRecognitionEstimator
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class FaceReidentificationConfig(Enum):
    Retail_0095_FP16_INT8 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp16-int8")
    Retail_0095_FP16 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp16")
    Retail_0095_FP32 = RepositoryAsset.openVino("face-reidentification-retail-0095-fp32")


class FaceReidentificationEstimator(FaceRecognitionEstimator):
    def __init__(self, model: Asset, weights: Asset, device: str = "AUTO"):
        super().__init__()
        self.engine = VisionInferenceEngine(model, weights, flip_channels=True, device=device)

        # left eye, right eye, tip of nose, left lip corner, right lip corner
        # https://docs.openvino.ai/latest/omz_models_model_face_reidentification_retail_0095.html
        self.normalized_keypoints = np.array([[0.31556875000000000, 0.4615741071428571],
                                              [0.68262291666666670, 0.4615741071428571],
                                              [0.50026249999999990, 0.6405053571428571],
                                              [0.34947187500000004, 0.8246919642857142],
                                              [0.65343645833333330, 0.8246919642857142]
                                              ], dtype=np.float32)

    def setup(self):
        self.engine.setup()

    def process(self, image: np.ndarray, landmarks: Optional[FaceLandmarkResult] = None) -> EmbeddingResult:
        image, landmarks = self._pre_process_input(image, landmarks)
        aligned_face, landmark_overlap = self._align_face(image, landmarks, self.normalized_keypoints)

        result = self.engine.process(aligned_face)
        data = result[self.engine.output_names[0]]
        flat_data = data.reshape((data.shape[1]))

        return EmbeddingResult(flat_data, landmark_overlap)

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    @staticmethod
    def create(config: FaceReidentificationConfig = FaceReidentificationConfig.Retail_0095_FP32) -> \
            "FaceReidentificationEstimator":
        model, weights = config.value
        return FaceReidentificationEstimator(model, weights)
