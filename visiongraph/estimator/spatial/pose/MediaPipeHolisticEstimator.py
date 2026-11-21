from enum import Enum

import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.result.spatial.pose.HolisticPose import HolisticPose
from visiongraph.util.MediaPipeUtils import _confidence_mask_to_uint8, mediapipe_landmarks_to_score_and_vector4d
from visiongraph.util.VectorUtils import list_of_vector4D


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe Holistic Landmarker model",
    source_url="https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker#models",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
    comment="Google confirms Apache 2.0 for MediaPipe models unless stated otherwise: "
    "https://github.com/google-ai-edge/mediapipe/issues/6355#issuecomment-5997480830",
)


class MediaPipeHolisticConfig(Enum):
    """Model assets for MediaPipe Holistic Landmarker."""

    Default = RepositoryAsset("holistic_landmarker.task", metadata=_MODEL_METADATA)


class MediaPipeHolisticEstimator(PoseEstimator[HolisticPose]):
    """Estimate body, face, and hand landmarks together with MediaPipe Holistic Landmarker."""

    def __init__(
        self,
        min_score: float = 0.5,
        static_image_mode: bool = False,
        enable_segmentation: bool = False,
        *,
        min_pose_presence_confidence: float = 0.5,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
        task: Asset = MediaPipeHolisticConfig.Default.value,
    ) -> None:
        """
        Initialize the holistic estimator.

        :param min_score: The minimum pose detection confidence.
        :param static_image_mode: Whether to estimate landmarks independently in each image.
        :param enable_segmentation: Whether to return a pose segmentation mask.
        :param min_pose_presence_confidence: The minimum pose presence confidence.
        :param min_face_detection_confidence: The minimum face detection confidence.
        :param min_face_presence_confidence: The minimum face presence confidence.
        :param min_hand_landmarks_confidence: The minimum hand landmarks confidence.
        :param task: MediaPipe task asset to use for holistic estimation.
        """
        super().__init__(min_score)
        self.static_image_mode = static_image_mode
        self.enable_segmentation = enable_segmentation
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_hand_landmarks_confidence = min_hand_landmarks_confidence
        self.task = task
        self._task = MediaPipeTaskRunner(
            vision.HolisticLandmarkerOptions,
            vision.HolisticLandmarker.create_from_options,
            vision.HolisticLandmarker.detect,
            vision.HolisticLandmarker.detect_for_video,
        )

    def _options(self) -> dict[str, vision.RunningMode | float | bool]:
        return dict(
            running_mode=vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_score,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_hand_landmarks_confidence=self.min_hand_landmarks_confidence,
            output_pose_segmentation_masks=self.enable_segmentation,
        )

    def setup(self) -> None:
        """Set up the holistic landmarker."""
        self._task.setup(self.task, **self._options())

    def process(self, data: np.ndarray, timestamp_ms: int | None = None) -> ResultList[HolisticPose]:
        """
        Estimate a pose and its available face, hands, and mask in a BGR frame.

        Option changes take effect on the next frame and reset tracking.
        """
        results = self._task.process(data, self.task, timestamp_ms, **self._options())
        if not results.pose_landmarks:
            return ResultList()

        score, landmarks = mediapipe_landmarks_to_score_and_vector4d(results.pose_landmarks)
        pose = HolisticPose(score, landmarks)

        if results.face_landmarks:
            face = list_of_vector4D([(point.x, point.y, point.z, 1.0) for point in results.face_landmarks])
            pose.face = BlazeFaceMesh(1.0, face)
        if results.right_hand_landmarks:
            right_hand = list_of_vector4D([(point.x, point.y, point.z, 1.0) for point in results.right_hand_landmarks])
            pose.right_hand = BlazeHand(1.0, right_hand, Handedness.RIGHT)
        if results.left_hand_landmarks:
            left_hand = list_of_vector4D([(point.x, point.y, point.z, 1.0) for point in results.left_hand_landmarks])
            pose.left_hand = BlazeHand(1.0, left_hand, Handedness.LEFT)
        if results.segmentation_mask is not None:
            pose.segmentation_mask = _confidence_mask_to_uint8(results.segmentation_mask.numpy_view())

        return ResultList([pose])

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()

    @staticmethod
    def create() -> "MediaPipeHolisticEstimator":
        """Create a holistic estimator with the default bundled model."""
        return MediaPipeHolisticEstimator()
