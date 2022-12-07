import copy
from typing import List, TypeVar, Tuple

import cv2
import numpy as np

from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util import ImageUtils

ODR = TypeVar("ODR", bound=ObjectDetectionResult)


def non_maximum_suppression(results: List[ODR], min_score: float, iou_threshold: float) -> List[ODR]:
    boxes = [list(result.bounding_box) for result in results]
    confidences = [result.score for result in results]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_score, iou_threshold)
    return [results[int(i)] for i in list(indices)]


def extract_object_detection_roi(image: np.ndarray,
                                 detection: ODR) -> Tuple[np.ndarray, ODR]:
    box: BoundingBox2D = detection.bounding_box.scale_with(Size2D.from_image(image))
    roi = ImageUtils.roi(image, box)

    result = copy.deepcopy(detection)
    result.map_coordinates(Size2D.from_image(image), Size2D.from_image(roi), src_roi=box)
    return roi, result
