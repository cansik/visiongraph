from enum import Enum
from typing import List

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
                 width: int, height: int, min_score: float = 0.5, nms_threshold: float = 0.5, device: str = "CPU"):
        super().__init__(min_score)
        self.labels = labels
        self.nms_threshold = nms_threshold

        self.engine = VisionInferenceEngine(model, weights, 1, 3, width, height,
                                            flip_channels=True, normalize=True, device=device)

    def setup(self):
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
