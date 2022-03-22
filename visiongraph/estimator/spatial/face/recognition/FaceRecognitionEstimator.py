from abc import abstractmethod, ABC
from typing import Optional, Tuple

import cv2
import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.SpatialCascadeResult import SpatialCascadeResult
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult
from visiongraph.result.spatial.face.RegressionFace import RegressionFace


class FaceRecognitionEstimator(RoiEstimator, ABC):
    def __init__(self):

        self.landmarks_key = "landmarks"
        self._landmarks: Optional[FaceLandmarkResult] = None

    def process_detection(self, image: np.ndarray,
                          detection: SpatialCascadeResult, rectified: bool = True) -> EmbeddingResult:

        if self.landmarks_key not in detection.results:
            raise Exception(f"Expecting landmarks in key '{self.landmarks_key}'")

        landmark_result: FaceLandmarkResult = detection.results["landmarks"]

        image_box = BoundingBox2D.from_image(image)
        self._landmarks = landmark_result.map(image_box,
                                              detection.bounding_box.scale(image_box.width, image_box.height))

        return super().process_detection(image, detection, rectified)

    @abstractmethod
    def process(self, image: np.ndarray,
                landmarks: Optional[FaceLandmarkResult] = None) -> EmbeddingResult:
        pass

    def _pre_process_input(self, data: np.ndarray,
                           landmarks: Optional[FaceLandmarkResult] = None) -> Tuple[np.ndarray, FaceLandmarkResult]:

        if landmarks is None:
            landmarks = self._landmarks

        return data, landmarks

    def _align_face(self, image: np.ndarray,
                    landmarks: FaceLandmarkResult,
                    normalized_keypoints: np.ndarray) -> Tuple[np.ndarray, float]:
        # align face
        src_keypoints = np.array([
            [landmarks.left_eye.x, landmarks.left_eye.y],
            [landmarks.right_eye.x, landmarks.right_eye.y],
            [landmarks.nose.x, landmarks.nose.y]
        ], dtype=np.float32)

        # use all landmarks if possible
        if isinstance(landmarks, RegressionFace):
            src_keypoints = np.vstack((src_keypoints,
                                       np.array([
                                           [landmarks.mouth_left.x, landmarks.mouth_left.y],
                                           [landmarks.mouth_right.x, landmarks.mouth_right.y]
                                       ], dtype=np.float32)
                                       ))

        scale = np.array((image.shape[1], image.shape[0]))
        desired_landmarks = np.array(normalized_keypoints[:src_keypoints.shape[0]], dtype=np.float64)
        landmarks = src_keypoints

        landmark_overlap = np.sqrt((np.power(desired_landmarks - landmarks, 2)).sum(axis=-1)).sum()

        transform = self._get_transform(desired_landmarks * scale, landmarks * scale)
        warped_image = cv2.warpAffine(image, transform, tuple(scale), flags=cv2.WARP_INVERSE_MAP)

        return warped_image, float(landmark_overlap)

    @staticmethod
    def _normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def _get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            '2d input arrays are expected, got {}'.format(src.shape)
        src_col_mean, src_col_std = FaceRecognitionEstimator._normalize(src, axis=0)
        dst_col_mean, dst_col_std = FaceRecognitionEstimator._normalize(dst, axis=0)

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform
