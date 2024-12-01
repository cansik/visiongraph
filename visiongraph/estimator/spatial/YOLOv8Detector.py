from enum import Enum

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class YOLOv8Config(Enum):
    """
    An enumeration class that defines YOLOv8 model configurations with their corresponding ONNX models and labels.
    """
    YOLOv8_N = RepositoryAsset("yolov8n.onnx"), COCO_80_LABELS
    YOLOv8_S = RepositoryAsset("yolov8s.onnx"), COCO_80_LABELS
    YOLOv8_M = RepositoryAsset("yolov8m.onnx"), COCO_80_LABELS
    YOLOv8_L = RepositoryAsset("yolov8l.onnx"), COCO_80_LABELS
    YOLOv8_X = RepositoryAsset("yolov8x.onnx"), COCO_80_LABELS


class YOLOv8Detector(UltralyticsYOLODetector[ObjectDetectionResult]):
    """
    A class representing a YOLOv8 object detector that specializes in detecting objects using UltralyticsYOLO framework.
    """

    @staticmethod
    def create(config: YOLOv8Config = YOLOv8Config.YOLOv8_S) -> "YOLOv8Detector":
        """
        Static method to create an instance of YOLOv8Detector based on the provided configuration.

        :param config: The configuration setting for the YOLOv8 model (default is YOLOv8_S).

        :return: A YOLOv8Detector instance initialized with the specified model and labels.
        """
        model, labels = config.value
        return YOLOv8Detector(model, labels=labels)
