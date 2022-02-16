from argparse import Namespace
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.BlazePoseSegmentation import BlazePoseSegmentation
from visiongraph.util.VectorUtils import list_of_vector4D

_mp_pose = mp.solutions.pose


class PoseModelComplexity(Enum):
    Light = 0
    Normal = 1
    Heavy = 2


class MediaPipePoseEstimator(PoseEstimator[BlazePose]):
    def __init__(self, complexity: PoseModelComplexity = PoseModelComplexity.Normal,
                 min_score: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True):
        super().__init__(min_score)

        self.smooth_landmarks = smooth_landmarks
        self.static_image_mode = static_image_mode
        self.min_tracking_confidence = min_tracking_confidence
        self.complexity = complexity

        self.smooth_segmentation = smooth_segmentation
        self.enable_segmentation = enable_segmentation

        self.detector: Optional[_mp_pose.Pose] = None

    def setup(self):
        self.detector = _mp_pose.Pose(static_image_mode=self.static_image_mode,
                                      model_complexity=self.complexity.value,
                                      min_detection_confidence=self.min_score,
                                      min_tracking_confidence=self.min_tracking_confidence,
                                      enable_segmentation=self.enable_segmentation,
                                      smooth_segmentation=self.smooth_segmentation)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
        # pre-process image
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image)

        # check if results are there
        if not results.pose_landmarks:
            return ResultList()

        raw_landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
        landmarks = list_of_vector4D(raw_landmarks)
        score = float(np.average(landmarks["t"]))

        if not self.enable_segmentation:
            return ResultList([BlazePose(score, landmarks)])

        # use segmentation
        mask = results.segmentation_mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        return ResultList([BlazePoseSegmentation(score, landmarks, mask_uint8)])

    def release(self):
        self.detector.close()

    def configure(self, args: Namespace):
        super().configure(args)
        # todo: implement arg parse

    @staticmethod
    def create(complexity: PoseModelComplexity = PoseModelComplexity.Normal) -> "MediaPipePoseEstimator":
        return MediaPipePoseEstimator(complexity)
