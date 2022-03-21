from argparse import ArgumentParser, Namespace
from typing import Optional

import cv2
import numpy as np

from visiongraph.estimator.spatial.face.recognition.FaceRecognitionEstimator import FaceRecognitionEstimator
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult
from visiongraph.result.spatial.face.RegressionFace import RegressionFace
from visiongraph.util import ImageUtils


class FaceReidentificationEstimator(FaceRecognitionEstimator):
    def __init__(self):
        super().__init__()

        # left eye, right eye, tip of nose, left lip corner, right lip corner
        # https://docs.openvino.ai/latest/omz_models_model_face_reidentification_retail_0095.html
        self.normalized_keypoints = np.array([[0.31556875000000000, 0.4615741071428571],
                                              [0.68262291666666670, 0.4615741071428571],
                                              [0.50026249999999990, 0.6405053571428571],
                                              [0.34947187500000004, 0.8246919642857142],
                                              [0.65343645833333330, 0.8246919642857142]
                                              ], dtype=np.float32)

    def setup(self):
        pass

    def process(self, image: np.ndarray, landmarks: Optional[FaceLandmarkResult] = None) -> EmbeddingResult:
        image, landmarks = self._pre_process_input(image, landmarks)
        aligned_face = self._align_face(image, landmarks, self.normalized_keypoints)

        

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
