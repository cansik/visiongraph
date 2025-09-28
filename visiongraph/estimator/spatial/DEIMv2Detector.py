from enum import Enum
from typing import List, Optional, Self

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.model.NMSOptions import NMSOptions
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression_from_options


class DEIMv2Config(Enum):
    """
    Enumeration of available DEIMv2 model configurations with their associated assets and label sets.
    """

    DEIMv2_HgNetv2_Atto_COCO = RepositoryAsset("deimv2_hgnetv2_atto_coco.onnx"), COCO_80_LABELS
    DEIMv2_HgNetv2_Femto_COCO = RepositoryAsset("deimv2_hgnetv2_femto_coco.onnx"), COCO_80_LABELS
    DEIMv2_HgNetv2_Pico_COCO = RepositoryAsset("deimv2_hgnetv2_pico_coco.onnx"), COCO_80_LABELS
    DEIMv2_HgNetv2_N_COCO = RepositoryAsset("deimv2_hgnetv2_n_coco.onnx"), COCO_80_LABELS

    DEIMv2_Dino3_S_COCO = RepositoryAsset("deimv2_dinov3_s_coco.onnx"), COCO_80_LABELS
    DEIMv2_Dino3_M_COCO = RepositoryAsset("deimv2_dinov3_m_coco.onnx"), COCO_80_LABELS
    DEIMv2_Dino3_L_COCO = RepositoryAsset("deimv2_dinov3_l_coco.onnx"), COCO_80_LABELS
    DEIMv2_Dino3_X_COCO = RepositoryAsset("deimv2_dinov3_x_coco.onnx"), COCO_80_LABELS


class DEIMv2Detector(ObjectDetector):
    """
    DEIMv2Detector is an object detector that uses the DEIMv2 models for spatial inference on images.
    Supports configurable model assets, label sets, inference engine, and post-processing with NMS.
    """

    def __init__(
        self,
        *assets: Asset,
        labels: List[str],
        min_score: float = 0.3,
        batch_size: int = 1,
        nms_options: Optional[NMSOptions] = None,
        engine: InferenceEngine = InferenceEngine.ONNX,
    ):
        """
        Initialize the DEIMv2Detector object with specified parameters.

        :param *assets: Variable length argument list of model assets.
        :param labels: List of label names corresponding to detection classes.
        :param min_score: Minimum confidence score threshold for detections (default is 0.3).
        :param batch_size: Number of images to process in a batch (default is 1).
        :param nms_options: Configuration for non-maximum suppression.
        :param engine: The inference engine to use (default is ONNX).
        """
        super().__init__(min_score)
        self.engine = InferenceEngineFactory.create(engine, assets, flip_channels=True, scale=255.0, padding=True)
        # set padding color
        self.engine.padding_color = (125, 125, 125)

        self.batch_size = batch_size
        self.labels: List[str] = labels
        self.nms_options = nms_options or NMSOptions(enabled=False)

    def setup(self):
        """
        Prepare the internal inference engine for processing.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[ObjectDetectionResult]:
        """
        Run object detection on an input image and return structured detection results.

        :param image: Input image as a NumPy array.

        :returns: A ResultList containing ObjectDetectionResult items.
        """
        h, w = self.engine.first_input_shape[2:]

        output = self.engine.process(image, {"orig_target_sizes": [[w, h]]})
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        boxes = boxes[0]
        labels = labels[0]
        scores = scores[0]

        # filter detection min score
        output_indices = np.where(scores > self.min_score)[0]

        # create result list
        results = ResultList()
        for i in output_indices:
            label = labels[i]
            if label < 0:
                continue

            box = boxes[i]
            score = scores[i]

            x1, y1, x2, y2 = box

            # find label
            label_index = label

            # process bounding box
            bbox = BoundingBox2D(x1, y1, x2 - x1, y2 - y1).scale(1 / w, 1 / h)

            detection = ObjectDetectionResult(label_index, self.labels[label_index], score, bbox)
            detection.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            results.append(detection)

        if self.nms_options.enabled:
            results = ResultList(non_maximum_suppression_from_options(results, self.nms_options))
        return results

    def release(self):
        """
        Free resources used by the inference engine.
        """
        self.engine.release()

    @classmethod
    def create(cls, config: DEIMv2Config = DEIMv2Config.DEIMv2_HgNetv2_Pico_COCO) -> Self:
        """
        Create a DEIMv2Detector instance from a predefined DEIMv2Config.

        :param config: DEIMv2Config enum value specifying model asset and labels.

        :returns: An initialized DEIMv2Detector instance.
        """
        asset, labels = config.value

        return DEIMv2Detector(asset, labels=labels)
