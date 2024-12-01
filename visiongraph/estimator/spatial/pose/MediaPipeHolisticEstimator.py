from argparse import Namespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.types.MediaPipePoseModelComplexity import PoseModelComplexity
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.HolisticPose import HolisticPose
from visiongraph.util.MediaPipeUtils import mediapipe_landmarks_to_score_and_vector4d, mediapipe_landmarks_to_vector4d

_mp_holistic = mp.solutions.holistic


class MediaPipeHolisticEstimator(PoseEstimator[HolisticPose]):
    """
    Estimates holistic poses using MediaPipe framework.
    """

    def __init__(self, complexity: PoseModelComplexity = PoseModelComplexity.Normal,
                 min_score: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 refine_landmarks: bool = True):
        """
        Initializes the MediaPipeHolisticEstimator with specified parameters.

        :param complexity: The complexity of the pose model to be used.
        :param min_score: Minimum score threshold for detection.
        :param min_tracking_confidence: Minimum confidence threshold for tracking.
        :param static_image_mode: Whether to use static image mode.
        :param smooth_landmarks: Whether to smooth landmarks.
        :param enable_segmentation: Whether to enable segmentation.
        :param smooth_segmentation: Whether to smooth segmentation results.
        :param refine_landmarks: Whether to refine landmarks.
        """
        super().__init__(min_score)

        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode
        self.min_tracking_confidence = min_tracking_confidence
        self.complexity = complexity

        self.smooth_segmentation = smooth_segmentation
        self.enable_segmentation = enable_segmentation

        self.refine_landmarks = refine_landmarks

        self.detector: Optional[_mp_holistic.Holistic] = None

    def setup(self):
        """
        Sets up the MediaPipe Holistic detector with the specified parameters.
        """
        self.detector = _mp_holistic.Holistic(static_image_mode=self.static_image_mode,
                                              model_complexity=self.complexity.value,
                                              min_detection_confidence=self.min_score,
                                              min_tracking_confidence=self.min_tracking_confidence,
                                              enable_segmentation=self.enable_segmentation,
                                              smooth_segmentation=self.smooth_segmentation,
                                              refine_face_landmarks=self.refine_landmarks)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
        """
        Processes an input image to extract pose landmarks and additional features.

        :param data: Input image in BGR format.

        :return: List of detected poses with associated features.
        """
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.detector.process(image)
        image.flags.writeable = True

        # check if results are there
        if not results.pose_landmarks:
            return ResultList()

        # create landmarks
        pose_score, pose_landmarks = mediapipe_landmarks_to_score_and_vector4d(results.pose_landmarks.landmark)
        pose = HolisticPose(pose_score, pose_landmarks)

        if results.face_landmarks:
            face_landmarks = mediapipe_landmarks_to_vector4d(results.face_landmarks.landmark)
            face_landmarks.t[:] = 1.0
            pose.face = BlazeFaceMesh(1.0, face_landmarks)

        if results.right_hand_landmarks:
            rh_landmarks = mediapipe_landmarks_to_vector4d(results.right_hand_landmarks.landmark)
            rh_landmarks.t[:] = 1.0
            pose.right_hand = BlazeHand(1.0, rh_landmarks, Handedness.RIGHT)

        if results.left_hand_landmarks:
            lh_landmarks = mediapipe_landmarks_to_vector4d(results.left_hand_landmarks.landmark)
            lh_landmarks.t[:] = 1.0
            pose.left_hand = BlazeHand(1.0, lh_landmarks, Handedness.LEFT)

        # use segmentation
        if results.segmentation_mask:
            mask = results.segmentation_mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            pose.segmentation_mask = mask_uint8

        return ResultList([pose])

    def release(self):
        """
        Releases the resources used by the MediaPipe Holistic detector.
        """
        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator with command-line arguments.

        :param args: Namespace containing configuration parameters.
        """
        super().configure(args)

    @staticmethod
    def create(complexity: PoseModelComplexity = PoseModelComplexity.Normal) -> "MediaPipeHolisticEstimator":
        """
        Creates an instance of MediaPipeHolisticEstimator.

        :param complexity: The complexity of the pose model to be used.

        :return: Instance of the estimator.
        """
        return MediaPipeHolisticEstimator(complexity)
