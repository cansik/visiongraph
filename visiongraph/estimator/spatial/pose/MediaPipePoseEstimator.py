from argparse import Namespace
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.BlazePoseSegmentation import BlazePoseSegmentation
from visiongraph.util.TimeUtils import HighPrecisionTimer
from visiongraph.util.VectorUtils import list_of_vector4D


class MediaPipePoseConfig(Enum):
    Light = RepositoryAsset("pose_landmarker_lite.task")
    Full = RepositoryAsset("pose_landmarker_full.task")
    Heavy = RepositoryAsset("pose_landmarker_heavy.task")


class MediaPipePoseEstimator(PoseEstimator[BlazePose]):

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_poses: int = 1,
                 min_pose_detection_confidence: float = 0.5,
                 min_pose_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 output_segmentation_masks: bool = True,
                 task: Asset = MediaPipePoseConfig.Full):
        super().__init__(min_pose_detection_confidence)

        self.static_image_mode = static_image_mode
        self.max_num_poses = max_num_poses
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.output_segmentation_masks = output_segmentation_masks

        self.detector: Optional[vision.PoseLandmarker] = None

        self.task = task

        self.timer = HighPrecisionTimer(ensure_monotonic=True)

    def setup(self):
        running_mode = VisionTaskRunningMode.IMAGE if self.static_image_mode else VisionTaskRunningMode.VIDEO

        # buffer loading because of https://github.com/google-ai-edge/mediapipe/issues/5343
        base_options = python.BaseOptions(model_asset_buffer=open(self.task.path, "rb").read())
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=self.max_num_poses,
            min_pose_detection_confidence=self.min_score,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, image: np.ndarray, timestamp_ms: Optional[int] = None) -> ResultList[BlazePose]:
        # Pre-process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run inference
        if self.static_image_mode:
            results = self.detector.detect(input_frame)
        else:
            if timestamp_ms is None:
                timestamp_ms = int(self.timer.time_ms())

            results = self.detector.detect_for_video(input_frame, timestamp_ms=timestamp_ms)

        if len(results.pose_landmarks) == 0:
            return ResultList()

        poses: ResultList[BlazePose] = ResultList()
        for i, pose_landmarks in enumerate(results.pose_landmarks):
            landmarks = [(rkp.x, rkp.y, rkp.z, rkp.visibility) for rkp in pose_landmarks]
            vector_landmarks = list_of_vector4D(landmarks)
            score = vector_landmarks.t.mean()

            if self.output_segmentation_masks:
                mask: np.ndarray = results.segmentation_masks[i].numpy_view()
                mask_uint8 = (mask * 255).astype(np.uint8)
                pose = BlazePoseSegmentation(score, vector_landmarks, mask_uint8)
            else:
                pose = BlazePose(score, vector_landmarks)

            poses.append(pose)

        return poses

    def release(self):
        self.detector.close()

    def configure(self, args: Namespace):
        super().configure(args)

    @staticmethod
    def create(config: MediaPipePoseConfig = MediaPipePoseConfig.Full, *args, **kwargs) -> "MediaPipePoseEstimator":
        return MediaPipePoseEstimator(*args, **kwargs, task=config.value)
