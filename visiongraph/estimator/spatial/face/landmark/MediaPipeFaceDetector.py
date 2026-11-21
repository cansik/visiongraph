from enum import Enum

import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFace import BlazeFace
from visiongraph.util.VectorUtils import list_of_vector4D


_SHORT_RANGE_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe BlazeFace (short range)",
    source_url="https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
)
_FULL_RANGE_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe BlazeFace Sparse (full range)",
    source_url="https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Sparse%20Model%20Card%20%28Full%20Range%29.pdf",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
)


class MediaPipeFaceModel(Enum):
    """Select a short- or full-range BlazeFace model."""

    Short_Range = 0
    Full_Range = 1


class MediaPipeFaceDetectorConfig(Enum):
    """BlazeFace model assets for MediaPipe Face Detector."""

    Short_Range = RepositoryAsset("blaze_face_short_range.tflite", metadata=_SHORT_RANGE_METADATA)
    Full_Range = RepositoryAsset("blaze_face_full_range_sparse.tflite", metadata=_FULL_RANGE_METADATA)


class MediaPipeFaceDetector(FaceLandmarkEstimator[BlazeFace]):
    """Detect faces and six keypoints per face with MediaPipe Face Detector."""

    def __init__(
        self,
        model: MediaPipeFaceModel = MediaPipeFaceModel.Short_Range,
        min_score: float = 0.5,
        *,
        task: Asset | None = None,
    ) -> None:
        """
        Initialize the face detector.

        :param model: The BlazeFace model preset.
        :param min_score: The minimum confidence for a face detection.
        :param task: A custom model asset that takes precedence over the preset.
        """
        super().__init__(min_score)
        self.model = model
        self.task = task
        self._task = MediaPipeTaskRunner(
            vision.FaceDetectorOptions,
            vision.FaceDetector.create_from_options,
            vision.FaceDetector.detect,
            vision.FaceDetector.detect_for_video,
        )

    def setup(self) -> None:
        """Set up the face detector."""
        asset = self.task if self.task is not None else MediaPipeFaceDetectorConfig[self.model.name].value
        self._task.setup(asset, running_mode=vision.RunningMode.IMAGE, min_detection_confidence=self.min_score)

    def process(self, image: np.ndarray) -> ResultList[BlazeFace]:
        """
        Detect faces in a BGR image.

        :return: Face detections with normalized bounding boxes and landmarks.
        """
        asset = self.task if self.task is not None else MediaPipeFaceDetectorConfig[self.model.name].value
        results = self._task.process(
            image, asset, running_mode=vision.RunningMode.IMAGE, min_detection_confidence=self.min_score
        )
        height, width = image.shape[:2]
        faces: ResultList[BlazeFace] = ResultList()
        for detection in results.detections:
            bounds = detection.bounding_box
            box = BoundingBox2D(
                bounds.origin_x / width, bounds.origin_y / height, bounds.width / width, bounds.height / height
            )
            landmarks = list_of_vector4D([(point.x, point.y, 0.0, 1.0) for point in detection.keypoints])
            faces.append(BlazeFace(detection.categories[0].score, landmarks, box))
        return faces

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()
