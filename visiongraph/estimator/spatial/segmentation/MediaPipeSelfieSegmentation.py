from enum import Enum

import numpy as np
from mediapipe.tasks.python import vision

from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner import MediaPipeTaskRunner
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.util.MediaPipeUtils import _confidence_mask_to_uint8


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="MediaPipe Selfie Segmenter models",
    source_url="https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf",
    license_name="Apache License 2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
)


class SelfieSegmentationModel(Enum):
    """Select a square or landscape selfie segmentation model."""

    General = 0
    Landscape = 1


class MediaPipeSelfieSegmentationConfig(Enum):
    """Selfie Segmenter model assets for MediaPipe Image Segmenter."""

    General = RepositoryAsset("selfie_segmenter.tflite", metadata=_MODEL_METADATA)
    Landscape = RepositoryAsset("selfie_segmenter_landscape.tflite", metadata=_MODEL_METADATA)


class MediaPipeSelfieSegmentation(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    """Estimate person segmentation masks with MediaPipe Image Segmenter."""

    def __init__(self, model_type: SelfieSegmentationModel = SelfieSegmentationModel.General) -> None:
        """
        Initialize the selfie segmenter.

        :param model_type: The selfie segmentation model preset.
        """
        super().__init__(0.5)
        self.model_type = model_type
        self._task = MediaPipeTaskRunner(
            vision.ImageSegmenterOptions,
            vision.ImageSegmenter.create_from_options,
            vision.ImageSegmenter.segment,
            vision.ImageSegmenter.segment_for_video,
        )

    def setup(self) -> None:
        """Set up the selfie segmenter."""
        asset = MediaPipeSelfieSegmentationConfig[self.model_type.name].value
        self._task.setup(asset, running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True)

    def process(self, data: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        """Return a frame-sized uint8 person confidence mask from a BGR image."""
        asset = MediaPipeSelfieSegmentationConfig[self.model_type.name].value
        result = self._task.process(data, asset, running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True)
        # Both Selfie Segmenter models output a single sigmoid mask for the person class.
        person_confidence = result.confidence_masks[0].numpy_view()
        mask = _confidence_mask_to_uint8(person_confidence)
        box = BoundingBox2D(0, 0, 1, 1)
        return ResultList([InstanceSegmentationResult(0, "human", 1.0, mask, box)])

    def release(self) -> None:
        """Release the task's resources."""
        self._task.release()

    @staticmethod
    def create(model_type: SelfieSegmentationModel = SelfieSegmentationModel.General) -> "MediaPipeSelfieSegmentation":
        """Create a selfie segmenter using the selected preset."""
        return MediaPipeSelfieSegmentation(model_type)
