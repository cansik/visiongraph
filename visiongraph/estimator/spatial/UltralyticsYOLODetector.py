from abc import ABC
from typing import List, Optional, TypeVar, Generic, Tuple

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression

R = TypeVar("R", bound=ObjectDetectionResult)


class UltralyticsYOLODetector(ObjectDetector, Generic[R], ABC):
    """
    A generic class representing an Ultralytics YOLO detector for object detection.

    Args:
        ObjectDetector: A class for detecting objects.
        Generic[R]: A generic type for object detection results.
        ABC: Abstract Base Class for defining abstract methods.
    """

    def __init__(self, *assets: Asset, labels: List[str], min_score: float = 0.3,
                 nms: bool = True, nms_threshold: float = 0.5,
                 nms_eta: Optional[float] = None, nms_top_k: Optional[int] = None,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        """
        Initializes the UltralyticsYOLODetector instance.

        Args:
            *assets (Asset): Variable number of assets.
            labels (List[str]): List of labels.
            min_score (float): Minimum score for predictions.
            nms (bool): Flag to enable non-maximum suppression.
            nms_threshold (float): Threshold for non-maximum suppression.
            nms_eta (Optional[float]): Epsilon value for non-maximum suppression.
            nms_top_k (Optional[int]): Top K value for non-maximum suppression.
            engine (InferenceEngine): Inference engine type.
        """
        super().__init__(min_score)
        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    scale=255.0,
                                                    padding=True)
        # set padding color
        self.engine.padding_color = (125, 125, 125)

        self.labels: List[str] = labels
        self.nms_threshold: float = nms_threshold
        self.nms: bool = nms
        self.nms_eta = nms_eta
        self.nms_top_k = nms_top_k

    def setup(self):
        """
        Sets up the Ultralytics YOLO detector.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[R]:
        """
        Processes the input image for object detection.

        Args:
            image (np.ndarray): Input image for detection.

        Returns:
            ResultList[R]: List of object detection results.
        """
        output = self.engine.process(image)

        predictions = output[self.engine.output_names[0]]
        predictions = np.squeeze(predictions)

        # filter detection min score
        predictions, scores = self._filter_predictions(predictions, self.min_score)

        # create result list
        results = ResultList()
        for pred, score in zip(predictions, scores):
            detection = self._decode_prediction(pred, score)
            detection.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            results.append(detection)

        if self.nms:
            results = ResultList(non_maximum_suppression(results, self.min_score, self.nms_threshold,
                                                         self.nms_eta, self.nms_top_k))
        return results

    def _filter_predictions(self, predictions: np.ndarray, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters the predictions based on the minimum score.

        Args:
            predictions (np.ndarray): Predicted values.
            min_score (float): Minimum score to consider.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered predictions and corresponding scores.
        """
        predictions = predictions.T

        scores = np.max(predictions[:, 4:4 + len(self.labels)], axis=1)
        valid_indices = scores > self.min_score
        predictions = predictions[valid_indices, :]
        return predictions, scores[valid_indices]

    def _unpack_box_prediction(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and unpacks box predictions.

        Args:
            prediction (np.ndarray): Predicted values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Unpacked box predictions.
        """
        return prediction[0:4], prediction[4:]

    def _decode_prediction(self, prediction: np.ndarray, score: float) -> R:
        """
        Decodes the prediction results to an ObjectDetectionResult.

        Args:
            prediction (np.ndarray): Predicted values.
            score (float): Prediction score.

        Returns:
            R: ObjectDetectionResult based on the prediction.
        """
        h, w = self.engine.first_input_shape[2:]
        pred_bbox, pred_label = self._unpack_box_prediction(prediction)

        # find label
        label_index = int(np.argmax(pred_label))

        # process bounding box
        wh = pred_bbox[2:]
        xy = pred_bbox[:2]
        xy -= wh * 0.5
        bbox = BoundingBox2D(float(xy[0]), float(xy[1]),
                             float(wh[0]), float(wh[1])).scale(1 / w, 1 / h)

        return ObjectDetectionResult(label_index, self.labels[label_index], float(score), bbox)

    def release(self):
        """
        Releases the resources used by the Ultralytics YOLO detector.
        """
        self.engine.release()
