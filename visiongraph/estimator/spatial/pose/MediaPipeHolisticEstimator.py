from argparse import Namespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.model.types.MediaPipePoseModelComplexity import PoseModelComplexity
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.BlazePoseSegmentation import BlazePoseSegmentation
from visiongraph.result.spatial.pose.HolisticPose import HolisticPose
from visiongraph.util.MediaPipeUtils import mediapipe_landmarks_to_score_and_vector4d, mediapipe_landmarks_to_vector4d
from visiongraph.util.VectorUtils import list_of_vector4D

_mp_holistic = mp.solutions.holistic


class MediaPipeHolisticEstimator(PoseEstimator[HolisticPose]):
    def __init__(self, complexity: PoseModelComplexity = PoseModelComplexity.Normal,
                 min_score: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 refine_landmarks: bool = True):
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
        self.detector = _mp_holistic.Holistic(static_image_mode=self.static_image_mode,
                                              model_complexity=self.complexity.value,
                                              min_detection_confidence=self.min_score,
                                              min_tracking_confidence=self.min_tracking_confidence,
                                              enable_segmentation=self.enable_segmentation,
                                              smooth_segmentation=self.smooth_segmentation,
                                              refine_face_landmarks=self.refine_landmarks)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
        # pre-process image
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.detector.process(image)
        image.flags.writeable = True

        # check if results are there
        if not results.pose_landmarks:
            return ResultList()

        # create landmarks
        pose_score, pose_landmarks = mediapipe_landmarks_to_score_and_vector4d(results.pose_landmarks.landmark)
        face_landmarks = mediapipe_landmarks_to_vector4d(results.face_landmarks.landmark)
        rh_landmarks = mediapipe_landmarks_to_vector4d(results.right_hand_landmarks.landmark)
        lh_landmarks = mediapipe_landmarks_to_vector4d(results.left_hand_landmarks.landmark)

        # fix scores
        face_landmarks.t[:] = 1.0
        rh_landmarks.t[:] = 1.0
        lh_landmarks.t[:] = 1.0

        pose = HolisticPose(pose_score, pose_landmarks,
                            1.0, face_landmarks,
                            1.0, rh_landmarks,
                            1.0, lh_landmarks)

        # use segmentation
        if self.enable_segmentation:
            mask = results.segmentation_mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            pose.segmentation_mask = mask_uint8

        return ResultList([pose])

    def release(self):
        self.detector.close()

    def configure(self, args: Namespace):
        super().configure(args)

    @staticmethod
    def create(complexity: PoseModelComplexity = PoseModelComplexity.Normal) -> "MediaPipeHolisticEstimator":
        return MediaPipeHolisticEstimator(complexity)
