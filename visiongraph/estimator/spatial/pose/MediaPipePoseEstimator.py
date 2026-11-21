from enum import Enum

import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.BlazePoseSegmentation import BlazePoseSegmentation
from visiongraph.util.MediaPipeUtils import _confidence_mask_to_uint8, mediapipe_landmarks_to_score_and_vector4d


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe pose landmarker tasks",
    source_url="https://github.com/google-ai-edge/mediapipe",
    license_name="Apache License 2.0",
    license_url="https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
)


class MediaPipePoseConfig(Enum):
    Light = RepositoryAsset("pose_landmarker_lite.task", metadata=_MODEL_METADATA)
    Full = RepositoryAsset("pose_landmarker_full.task", metadata=_MODEL_METADATA)
    Heavy = RepositoryAsset("pose_landmarker_heavy.task", metadata=_MODEL_METADATA)


class MediaPipePoseEstimator(PoseEstimator[BlazePose]):
    """Estimate body landmarks and optional segmentation masks with MediaPipe Pose Landmarker."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = True,
        task: Asset = MediaPipePoseConfig.Full.value,
    ) -> None:
        super().__init__(min_pose_detection_confidence)
        self.static_image_mode = static_image_mode
        self.max_num_poses = max_num_poses
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_segmentation_masks = output_segmentation_masks
        self.task = task
        self._task = MediaPipeTaskRunner(
            vision.PoseLandmarkerOptions,
            vision.PoseLandmarker.create_from_options,
            vision.PoseLandmarker.detect,
            vision.PoseLandmarker.detect_for_video,
        )

    def _options(self) -> dict[str, vision.RunningMode | int | float | bool]:
        return dict(
            running_mode=vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO,
            num_poses=self.max_num_poses,
            min_pose_detection_confidence=self.min_score,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks,
        )

    def setup(self) -> None:
        """Set up the pose landmarker."""
        self._task.setup(self.task, **self._options())

    def process(self, image: np.ndarray, timestamp_ms: int | None = None) -> ResultList[BlazePose]:
        """
        Estimate poses in a BGR image or video frame.

        Option changes take effect on the next frame and reset tracking.
        """
        results = self._task.process(image, self.task, timestamp_ms, **self._options())
        poses: ResultList[BlazePose] = ResultList()
        for index, pose_landmarks in enumerate(results.pose_landmarks):
            score, landmarks = mediapipe_landmarks_to_score_and_vector4d(pose_landmarks)
            if results.segmentation_masks is not None:
                mask = _confidence_mask_to_uint8(results.segmentation_masks[index].numpy_view())
                poses.append(BlazePoseSegmentation(score, landmarks, mask))
            else:
                poses.append(BlazePose(score, landmarks))
        return poses

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()

    @staticmethod
    def create(config: MediaPipePoseConfig = MediaPipePoseConfig.Full, *args, **kwargs) -> "MediaPipePoseEstimator":
        """Create a pose estimator using the selected model preset."""
        return MediaPipePoseEstimator(*args, **kwargs, task=config.value)
