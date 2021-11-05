from enum import Enum
from typing import List, Optional

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class YOLOConfig(Enum):
    YOLOv4_Tiny_VOC_256x256_FP32 = (*RepositoryAsset.openVino("yolov4_tiny_voc_256x256_FP32"), COCO_80_LABELS, 256)
    YOLOv4_Tiny_VOC_416x416_FP32 = (*RepositoryAsset.openVino("yolov4_tiny_voc_416x416_FP32"), COCO_80_LABELS, 416)


class YOLODetector(ObjectDetector):

    def __init__(self, model: Asset, weights: Asset, labels: List[str],
                 width: int, height: int, min_score: float = 0.5, iou_threshold: float = 0.5, device: str = "CPU"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.labels = labels
        self.width = width
        self.height = height
        self.iou_threshold = iou_threshold
        self.device = device

        self.engine: Optional[VisionInferenceEngine] = None

    def setup(self):
        self.engine = VisionInferenceEngine(self.model, self.weights, 1, 3, self.width, self.height,
                                            flip_channels=True, normalize=True, device=self.device)
        self.engine.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> List[ObjectDetectionResult]:
        output = self.engine.estimate(image)

        raise NotImplementedError("YOLO has to be implemented first.")
        # todo: implement yolo by using the OpenVino Example
        print(output)

    def release(self):
        self.engine.release()

    @staticmethod
    def create(config: YOLOConfig) -> "YOLODetector":
        model, weights, labels, size = config.value
        return YOLODetector(model, weights, labels, size, size)
