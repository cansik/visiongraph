from enum import Enum
from typing import List

import cv2
import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class YOLOModel(Enum):
    YOLOv4_Tiny_VOC_256x256_FP32 = ("yolov4_tiny_voc_256x256_FP32", 256, 256)
    YOLOv4_Tiny_VOC_416x416_FP32 = ("yolov4_tiny_voc_416x416_FP32", 416, 416)


class YOLODetector(ObjectDetector):

    def __init__(self, min_score: float = 0.5, nms_threshold: float = 0.5):
        super().__init__(min_score)

    def estimate(self, image: np.ndarray, **kwargs) -> List[ObjectDetectionResult]:
        pass

    def setup(self):
        pass

    def release(self):
        pass
