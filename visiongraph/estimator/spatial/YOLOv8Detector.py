from enum import Enum

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class YOLOv8Config(Enum):
    YOLOv8_N = RepositoryAsset("yolov8n.onnx"), COCO_80_LABELS
    YOLOv8_S = RepositoryAsset("yolov8s.onnx"), COCO_80_LABELS
    YOLOv8_M = RepositoryAsset("yolov8m.onnx"), COCO_80_LABELS
    YOLOv8_L = RepositoryAsset("yolov8l.onnx"), COCO_80_LABELS
    YOLOv8_X = RepositoryAsset("yolov8x.onnx"), COCO_80_LABELS


class YOLOv8Detector(UltralyticsYOLODetector[ObjectDetectionResult]):

    @staticmethod
    def create(config: YOLOv8Config = YOLOv8Config.YOLOv8_S) -> "YOLOv8Detector":
        model, labels = config.value
        return YOLOv8Detector(model, labels=labels)
