from typing import List, TypeVar

import cv2

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

ODR = TypeVar("ODR", bound=ObjectDetectionResult)


def non_maximum_suppression(results: List[ODR], min_score: float, iou_threshold: float) -> List[ODR]:
    boxes = [list(result.bounding_box) for result in results]
    confidences = [result.score for result in results]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_score, iou_threshold)
    return [results[i] for i in list(indices)]
