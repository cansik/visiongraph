from enum import Enum
from typing import Tuple

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector


class YOLOv5Config(Enum):
    YOLOv5_N = RepositoryAsset("yolov5n.onnx"), COCO_80_LABELS
    YOLOv5_S = RepositoryAsset("yolov5s.onnx"), COCO_80_LABELS
    YOLOv5_M = RepositoryAsset("yolov5m.onnx"), COCO_80_LABELS
    YOLOv5_L = RepositoryAsset("yolov5l.onnx"), COCO_80_LABELS
    YOLOv5_X = RepositoryAsset("yolov5x.onnx"), COCO_80_LABELS


class YOLOv5Detector(UltralyticsYOLODetector):

    def _filter_predictions(self, predictions: np.ndarray, min_score: float):
        valid_predictions = np.where(predictions[:, 4] > min_score)
        predictions = predictions[valid_predictions]
        scores = predictions[:, 4]
        return predictions, scores

    def _unpack_box_prediction(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return prediction[0:4], prediction[5:]

    @staticmethod
    def create(config: YOLOv5Config = YOLOv5Config.YOLOv5_S) -> "YOLOv5Detector":
        model, labels = config.value
        return YOLOv5Detector(model, labels=labels)
