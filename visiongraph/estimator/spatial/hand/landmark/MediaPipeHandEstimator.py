from enum import Enum

import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.hand.landmark.HandLandmarkEstimator import HandLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.util.VectorUtils import list_of_vector4D


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe Hand Landmarker model",
    source_url="https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
)


class MediaPipeHandConfig(Enum):
    """Model assets for MediaPipe Hand Landmarker."""

    Default = RepositoryAsset("hand_landmarker.task", metadata=_MODEL_METADATA)


class MediaPipeHandEstimator(HandLandmarkEstimator[BlazeHand]):
    """Estimate hand landmarks and handedness with MediaPipe Hand Landmarker."""

    def __init__(
        self,
        min_score: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        *,
        min_hand_presence_confidence: float = 0.5,
        task: Asset = MediaPipeHandConfig.Default.value,
    ) -> None:
        """
        Initialize the hand estimator.

        :param min_score: The minimum hand detection confidence.
        :param min_tracking_confidence: The minimum confidence for tracking hands.
        :param static_image_mode: Whether to detect hands independently in each image.
        :param max_num_hands: The maximum number of hands to detect.
        :param min_hand_presence_confidence: The minimum hand presence confidence.
        :param task: MediaPipe task asset to use for hand estimation.
        """
        super().__init__(min_score)
        self.static_image_mode = static_image_mode
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.task = task
        self._task = MediaPipeTaskRunner(
            vision.HandLandmarkerOptions,
            vision.HandLandmarker.create_from_options,
            vision.HandLandmarker.detect,
            vision.HandLandmarker.detect_for_video,
        )

    def _options(self) -> dict[str, vision.RunningMode | int | float]:
        return dict(
            running_mode=vision.RunningMode.IMAGE if self.static_image_mode else vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_score,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def setup(self) -> None:
        """Set up the hand landmarker."""
        self._task.setup(self.task, **self._options())

    def process(self, image: np.ndarray, timestamp_ms: int | None = None, **kwargs) -> ResultList[BlazeHand]:
        """
        Estimate hands in a BGR image or video frame.

        Option changes take effect on the next frame and reset tracking.
        """
        results = self._task.process(image, self.task, timestamp_ms, **self._options())
        hands: ResultList[BlazeHand] = ResultList()
        for landmarks, categories in zip(results.hand_landmarks, results.handedness):
            category = categories[0]
            handedness = Handedness.LEFT if category.category_name == "Left" else Handedness.RIGHT
            coordinates = list_of_vector4D([(point.x, point.y, point.z, 1.0) for point in landmarks])
            hands.append(BlazeHand(category.score, coordinates, handedness))
        return hands

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()
