from enum import Enum

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.data.labels.OpenImagesV7 import Open_Images_V7
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

    YOLOv8_N_Open_Images_V7 = RepositoryAsset("yolov8n-oiv7.onnx"), Open_Images_V7
    YOLOv8_S_Open_Images_V7 = RepositoryAsset("yolov8s-oiv7.onnx"), Open_Images_V7
    YOLOv8_M_Open_Images_V7 = RepositoryAsset("yolov8m-oiv7.onnx"), Open_Images_V7
    YOLOv8_L_Open_Images_V7 = RepositoryAsset("yolov8l-oiv7.onnx"), Open_Images_V7
    YOLOv8_X_Open_Images_V7 = RepositoryAsset("yolov8x-oiv7.onnx"), Open_Images_V7


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
