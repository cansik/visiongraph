from enum import Enum

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.DOTA import DOTA_v1_0
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.OrientedObjectDetectionResult import OrientedObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression


class YOLOv8OBBConfig(Enum):
    """
    An enumeration of YOLOv8 OBB model configurations.
    Each configuration includes the model asset and the dataset version.
    """
    YOLOv8_OBB_N = RepositoryAsset("yolov8n-obb.onnx"), DOTA_v1_0
    YOLOv8_OBB_S = RepositoryAsset("yolov8s-obb.onnx"), DOTA_v1_0
    YOLOv8_OBB_M = RepositoryAsset("yolov8m-obb.onnx"), DOTA_v1_0
    YOLOv8_OBB_L = RepositoryAsset("yolov8l-obb.onnx"), DOTA_v1_0
    YOLOv8_OBB_X = RepositoryAsset("yolov8x-obb.onnx"), DOTA_v1_0


class YOLOv8OBBDetector(UltralyticsYOLODetector[OrientedObjectDetectionResult]):
    """
    YOLOv8 Oriented Bounding Box Detector.
    Performs object detection and orients the detected bounding boxes.

    :param UltralyticsYOLODetector: The Ultralytics YOLO Object Detector class.
    """

    def process(self, image: np.ndarray) -> ResultList[OrientedObjectDetectionResult]:
        """
        Processes detection on the input image and returns a list of oriented object detection results.

        :param image: The input image as a NumPy array.

        :return: A list of oriented object detection results.
        """
        output = self.engine.process(image)

        predictions = output[self.engine.output_names[0]]
        predictions = np.squeeze(predictions)

        # filter detection min score
        predictions, scores = self._filter_predictions(predictions, self.min_score)

        if len(scores) == 0:
            return ResultList()

        class_ids = np.argmax(predictions[:, 4:4 + len(self.labels)], axis=1)
        boxes = predictions[:, :4]  # cxcywh
        angles = predictions[:, 4 + len(self.labels)]

        h, w = self.engine.first_input_shape[2:]

        # create result list
        results = ResultList()
        for score, class_id, box, theta in zip(scores, class_ids, boxes, angles):
            # process bounding box
            wh = box[2:4]
            xy = box[0:2]
            xy -= wh * 0.5
            bbox = BoundingBox2D(xy[0], xy[1], wh[0], wh[1]).scale(1 / w, 1 / h)

            # fix angle
            if np.pi <= theta <= 0.75 * np.pi:
                theta -= np.pi

            detection = OrientedObjectDetectionResult(class_id, self.labels[class_id], score, bbox, np.degrees(theta))
            detection.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            results.append(detection)

        if self.nms:
            results = ResultList(non_maximum_suppression(results, self.min_score, self.nms_threshold))
        return results

    @staticmethod
    def create(config: YOLOv8OBBConfig = YOLOv8OBBConfig.YOLOv8_OBB_S) -> "YOLOv8OBBDetector":
        """
        Instantiates a YOLOv8 Oriented Bounding Box Detector based on the provided configuration.

        :param config: The configuration for the detector. Defaults to YOLOv8_OBB_S.

        :return: An instance of YOLOv8 Oriented Bounding Box Detector.
        """
        model, labels = config.value
        return YOLOv8OBBDetector(model, labels=labels)
