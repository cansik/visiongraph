from argparse import Namespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.types.MediaPipePoseModelComplexity import PoseModelComplexity
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.BlazePoseSegmentation import BlazePoseSegmentation
from visiongraph.util.MediaPipeUtils import mediapipe_landmarks_to_score_and_vector4d

_mp_pose = mp.solutions.pose


class MediaPipePoseEstimatorLegacy(PoseEstimator[BlazePose]):
    """
    A pose estimator using MediaPipe to detect and process human poses in images.
    """

    def __init__(self, complexity: PoseModelComplexity = PoseModelComplexity.Normal,
                 min_score: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True):
        """
        Initializes the MediaPipePoseEstimator with specified parameters.

        :param complexity: Complexity of the pose model.
        :param min_score: Minimum score threshold for valid detections.
        :param min_tracking_confidence: Confidence threshold for tracking landmarks.
        :param static_image_mode: Indicates if the input images are static.
        :param smooth_landmarks: If True, applies smoothing to the detected landmarks.
        :param enable_segmentation: If True, enables segmentation for the pose.
        :param smooth_segmentation: If True, applies smoothing to the segmentation mask.
        """
        super().__init__(min_score)

        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode
        self.min_tracking_confidence = min_tracking_confidence
        self.complexity = complexity

        self.smooth_segmentation = smooth_segmentation
        self.enable_segmentation = enable_segmentation

        self.detector: Optional[_mp_pose.Pose] = None

    def setup(self):
        """
        Sets up the MediaPipe pose detector with the specified configuration.
        """
        self.detector = _mp_pose.Pose(static_image_mode=self.static_image_mode,
                                      model_complexity=self.complexity.value,
                                      min_detection_confidence=self.min_score,
                                      min_tracking_confidence=self.min_tracking_confidence,
                                      enable_segmentation=self.enable_segmentation,
                                      smooth_segmentation=self.smooth_segmentation)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
        """
        Processes an input image to detect poses and return results.

        :param data: The input image in BGR format.

        :return: A list of detected BlazePose objects.
        """
        # pre-process image
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image)

        # check if results are there
        if not results.pose_landmarks:
            return ResultList()

        score, landmarks = mediapipe_landmarks_to_score_and_vector4d(results.pose_landmarks.landmark)

        if not self.enable_segmentation:
            return ResultList([BlazePose(score, landmarks)])

        # use segmentation
        mask = results.segmentation_mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        return ResultList([BlazePoseSegmentation(score, landmarks, mask_uint8)])

    def release(self):
        """
        Releases resources used by the pose detector.
        """
        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator with command-line arguments.

        :param args: The command-line arguments.
        """
        super().configure(args)
        # todo: implement arg parse

    @staticmethod
    def create(complexity: PoseModelComplexity = PoseModelComplexity.Normal) -> "MediaPipePoseEstimatorLegacy":
        """
        Creates a new instance of MediaPipePoseEstimator with the specified complexity.

        :param complexity: The complexity level for the new instance.

        :return: A new instance of MediaPipePoseEstimator.
        """
        return MediaPipePoseEstimatorLegacy(complexity)
