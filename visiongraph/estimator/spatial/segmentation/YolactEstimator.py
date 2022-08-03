from enum import Enum
from typing import List

import cv2
import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.Asset import Asset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.BaseVisionEngine import PADDING_BOX_OUTPUT_NAME
from visiongraph.estimator.onnx.ONNXVisionEngine import ONNXVisionEngine
from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator, OutputType
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.util import MathUtils


class YolactConfig(Enum):
    YolactEdge_MobileNetV2_550 = (RepositoryAsset("yolact_edge_mobilenetv2_550x550.onnx"), COCO_80_LABELS)


class YolcatEstimator(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    def __init__(self, model: Asset, labels: List[str], min_score: float = 0.1):
        super().__init__(min_score)

        self.labels = labels
        # todo: enable padding for better results (not implemented yet)
        self.engine = ONNXVisionEngine(model, flip_channels=True, padding=False)

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[InstanceSegmentationResult]:
        ih, iw = data.shape[:2]
        outputs = self.engine.process(data)

        x1y1x2y2_score_class = outputs["x1y1x2y2_score_class"]
        final_masks = outputs["final_masks"]
        padding_box: BoundingBox2D = outputs[PADDING_BOX_OUTPUT_NAME]

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

            cropped = cv2.resize(cropped, (iw, ih))

            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            box = BoundingBox2D(
                MathUtils.map_value(x - padding_box.x_min, 0, padding_box.width, 0, 1),
                MathUtils.map_value(y - padding_box.y_min, 0, padding_box.height, 0, 1),
                w, h)

            results.append(InstanceSegmentationResult(class_id, self.labels[class_id], score, cropped, box))

        return results

    def release(self):
        self.engine.release()

    @staticmethod
    def _crop(bbox, shape):
        x1 = int(max(bbox[0] * shape[1], 0))
        y1 = int(max(bbox[1] * shape[0], 0))
        x2 = int(max(bbox[2] * shape[1], 0))
        y2 = int(max(bbox[3] * shape[0], 0))
        return slice(y1, y2), slice(x1, x2)

    @staticmethod
    def create(config: YolactConfig = YolactConfig.YolactEdge_MobileNetV2_550) -> "YolcatEstimator":
        model, labels = config.value
        return YolcatEstimator(model, labels)
