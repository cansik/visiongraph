import copy
from abc import abstractmethod, ABC
from typing import Optional, Tuple

import cv2
import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.SpatialCascadeResult import SpatialCascadeResult
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class FaceRecognitionEstimator(RoiEstimator, ABC):
    """
    An abstract class for face recognition estimators that process facial images and landmarks.

    Inherits from RoiEstimator to utilize region of interest (ROI) estimation functionality.
    """

    def __init__(self):
        """
        Initializes the FaceRecognitionEstimator with default attributes.
        """
        self.landmarks_key = "landmarks"
        self._landmarks: Optional[FaceLandmarkResult] = None

    def process_detection(self, image: np.ndarray,
                          detection: SpatialCascadeResult, rectified: bool = True) -> EmbeddingResult:
        """
        Processes a detected face in the image by extracting and mapping landmarks.

        :param image: The input image containing the face.
        :param detection: The result of a face detection containing bounding box and landmarks.
        :param rectified: A flag indicating if the image has been rectified.

        :return: The result of the embedding after processing the detection.

        :raises Exception: If the landmarks key is not present in the detection results.
        """
        if self.landmarks_key not in detection.results:
            raise Exception(f"Expecting landmarks in key '{self.landmarks_key}'")

        landmark_result: FaceLandmarkResult = copy.deepcopy(detection.results["landmarks"])

        image_size = Size2D.from_image(image)
        detection_box = detection.bounding_box.scale_with(image_size)
        landmark_result.map_coordinates(image_size, detection_box.size, src_roi=detection_box)

        self._landmarks = landmark_result

        return super().process_detection(image, detection, rectified)

    @abstractmethod
    def process(self, image: np.ndarray,
                landmarks: Optional[FaceLandmarkResult] = None) -> EmbeddingResult:
        """
        Processes the input image and landmarks to produce an embedding result.

        :param image: The input image to process.
        :param landmarks: Optional landmarks to be used during processing.

        :return: The result of the embedding after processing.
        """
        pass

    def _pre_process_input(self, data: np.ndarray,
                           landmarks: Optional[FaceLandmarkResult] = None) -> Tuple[np.ndarray, FaceLandmarkResult]:
        """
        Prepares the input data and landmarks for processing.

        :param data: The input data to preprocess.
        :param landmarks: Optional landmarks for the input.

        :return: The preprocessed data and associated landmarks.
        """
        if landmarks is None:
            landmarks = self._landmarks

        return data, landmarks

    def _align_face(self, image: np.ndarray,
                    landmarks: FaceLandmarkResult,
                    normalized_keypoints: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Aligns the face in the image based on the provided landmarks and normalized keypoints.

        :param image: The input image containing the face to be aligned.
        :param landmarks: The landmarks of the detected face.
        :param normalized_keypoints: The desired normalized keypoints for alignment.

        :return: The aligned image and the overlap score of the landmarks.
        """
        src_keypoints = np.array([
            [landmarks.left_eye.x, landmarks.left_eye.y],
            [landmarks.right_eye.x, landmarks.right_eye.y],
            [landmarks.nose.x, landmarks.nose.y]
        ], dtype=np.float32)

        # use all landmarks if possible
        if hasattr(landmarks, "mouth_left") and hasattr(landmarks, "mouth_right"):
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
        """
        Normalizes the input array along the specified axis.

        :param array: The input array to normalize.
        :param axis: The axis along which to normalize the array.

        :return: The mean and standard deviation used for normalization.
        """
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def _get_transform(src, dst):
        """
        Computes the transformation matrix to align two sets of points.

        :param src: The source points.
        :param dst: The destination points.

        :return: The transformation matrix for aligning the points.

        :raises AssertionError: If the input arrays are not 2D or do not have equal shapes.
        """
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
