from enum import Enum
from typing import List

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.onnx.ONNXVisionEngine import ONNXVisionEngine
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.util import ImageUtils


class YolactConfig(Enum):
    """
    Enum containing configuration options for the Yolcat estimator.
    """
    YolactEdge_MobileNetV2_550 = (RepositoryAsset("yolact_edge_mobilenetv2_550x550.onnx"), COCO_80_LABELS)


class YolcatEstimator(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    """
    An instance segmentation estimator using the Yolcat model.
    """

    def __init__(self, model: Asset, labels: List[str], min_score: float = 0.1):
        """
        Initializes the Yolcat estimator.

        :param model: The ONNX model to be used for inference.
        :param labels: A list of class names corresponding to each label index.
        :param min_score: The minimum score required for a detected instance to be considered. Defaults to 0.1.
        """
        super().__init__(min_score)

        self.labels = labels
        self.engine = ONNXVisionEngine(model, flip_channels=True, padding=True)

    def setup(self):
        """
        Sets up the estimator by calling the setup method of the underlying ONNX engine.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        """
        Processes the input image and returns a list of instance segmentation results.

        :param data: The input image to be processed.

        :return: A list of instance segmentation results.
        """
        ih, iw = data.shape[:2]
        outputs = self.engine.process(data)

        x1y1x2y2_score_class = outputs["x1y1x2y2_score_class"]
        final_masks = outputs["final_masks"]
        padding_box = outputs.padding_box.scale(1 / outputs.image_size.width, 1 / outputs.image_size.height)

        results = ResultList()

        for result, mask in zip(x1y1x2y2_score_class[0], final_masks):
            bbox = result[:4].tolist()
            score = result[4]
            class_id = int(result[5])

            if self.min_score > score:
                continue

            # Add 1 to class_id to distinguish it from the background 0
            mask = np.where(mask > 0.5, class_id + 1, 0).astype(np.uint8)
            region = self._crop(bbox, mask.shape)
            cropped = np.zeros(mask.shape, dtype=np.uint8)
            cropped[region] = mask[region]

            mask_box = padding_box.scale(mask.shape[1], mask.shape[0])
            cropped = ImageUtils.roi(cropped, mask_box)

            cropped = cv2.resize(cropped, (iw, ih))

            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            box = BoundingBox2D(x, y, w, h)

            result = InstanceSegmentationResult(class_id, self.labels[class_id], score, cropped, box)
            result.map_coordinates(Size2D.from_image(mask), (iw, ih), src_roi=mask_box)
            results.append(result)

        return results

    def release(self):
        """
        Releases the estimator's resources.
        """
        self.engine.release()

    @staticmethod
    def _crop(bbox, shape):
        """
        Crops a region of interest from an image.

        :param bbox: A list containing the coordinates of the top-left and bottom-right corners of the ROI.
        :param shape: The size of the input image.

        :return: A tuple containing two slice objects representing the cropped region.
        """
        x1 = int(max(bbox[0] * shape[1], 0))
        y1 = int(max(bbox[1] * shape[0], 0))
        x2 = int(max(bbox[2] * shape[1], 0))
        y2 = int(max(bbox[3] * shape[0], 0))
        return slice(y1, y2), slice(x1, x2)

    @staticmethod
    def create(config: YolactConfig = YolactConfig.YolactEdge_MobileNetV2_550) -> "YolcatEstimator":
        """
        Creates a new instance of the Yolcat estimator.

        :param config: The configuration option to be used for the estimator. Defaults to YolcatConfig.YolactEdge_MobileNetV2_550.
        """
        model, labels = config.value
        return YolcatEstimator(model, labels)
