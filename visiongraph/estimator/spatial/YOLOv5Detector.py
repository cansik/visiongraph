from enum import Enum
from typing import Tuple

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector


class YOLOv5Config(Enum):
    """
    An enumeration of YOLOv5 model configurations including the model asset and labels.
    """
    YOLOv5_N = RepositoryAsset("yolov5n.onnx"), COCO_80_LABELS
    YOLOv5_S = RepositoryAsset("yolov5s.onnx"), COCO_80_LABELS
    YOLOv5_M = RepositoryAsset("yolov5m.onnx"), COCO_80_LABELS
    YOLOv5_L = RepositoryAsset("yolov5l.onnx"), COCO_80_LABELS
    YOLOv5_X = RepositoryAsset("yolov5x.onnx"), COCO_80_LABELS


class YOLOv5Detector(UltralyticsYOLODetector):
    """
    A detector class for YOLOv5 models based on UltralyticsYOLODetector.
    """

    def _filter_predictions(self, predictions: np.ndarray, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters the predictions based on a minimum score.

        :param predictions: An array of predictions to filter.
        :param min_score: The minimum score threshold for valid predictions.

        :return: A tuple of filtered predictions and corresponding scores.
        """
        valid_predictions = np.where(predictions[:, 4] > min_score)
        predictions = predictions[valid_predictions]
        scores = predictions[:, 4]
        return predictions, scores

    def _unpack_box_prediction(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpacks the box prediction.

        :param prediction: The prediction to unpack.

        :return: A tuple of the box coordinates and additional data.
        """
        return prediction[0:4], prediction[5:]

    @staticmethod
    def create(config: YOLOv5Config = YOLOv5Config.YOLOv5_S) -> "YOLOv5Detector":
        """
        Creates a YOLOv5Detector object based on a given configuration.

        :param config: The configuration for the YOLOv5 model.

        :return: A YOLOv5Detector object initialized with the specified configuration.
        """
        model, labels = config.value
        return YOLOv5Detector(model, labels=labels)
