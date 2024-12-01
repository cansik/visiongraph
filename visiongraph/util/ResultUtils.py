import copy
from typing import List, TypeVar, Tuple, Optional

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util import ImageUtils

ODR = TypeVar("ODR", bound=ObjectDetectionResult)


def non_maximum_suppression(results: List[ODR], min_score: float, iou_threshold: float,
                            eta: Optional[float] = None, top_k: Optional[int] = None) -> List[ODR]:
    """
    Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.

    :param results: List of object detection results.
    :param min_score: Minimum score threshold to consider a box.
    :param iou_threshold: IOU threshold for merging boxes.
    :param eta: Optional parameter for adjusting NMS.
    :param top_k: Optional parameter to limit the number of boxes.

    :return: List of filtered object detection results after NMS.
    """
    boxes = [list(result.bounding_box) for result in results]
    confidences = [result.score for result in results]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_score, iou_threshold, eta, top_k)
    return [results[int(i)] for i in list(indices)]


def extract_object_detection_roi(image: np.ndarray,
                                 detection: ODR) -> Tuple[np.ndarray, ODR]:
    """
    Extracts the region of interest (ROI) from an image based on the detected bounding box.

    :param image: Input image from which to extract the ROI.
    :param detection: Object detection result containing the bounding box.

    :return: A tuple containing the extracted ROI and the modified detection result.
    """
    box: BoundingBox2D = detection.bounding_box.scale_with(Size2D.from_image(image))
    roi = ImageUtils.roi(image, box)

    result = copy.deepcopy(detection)
    result.map_coordinates(Size2D.from_image(image), Size2D.from_image(roi), src_roi=box)
    return roi, result


def bbox_from_landmarks(landmarks: vector.VectorNumpy4D) -> BoundingBox2D:
    """
    Creates a bounding box from a set of landmarks.

    :param landmarks: A set of landmarks containing 'x' and 'y' coordinates.

    :return: The bounding box encompassing the provided landmarks.
    """
    xs = np.ma.masked_equal(landmarks["x"], 0.0, copy=False)
    ys = np.ma.masked_equal(landmarks["y"], 0.0, copy=False)

    x_min = xs.min()
    y_min = ys.min()
    x_max = xs.max()
    y_max = ys.max()

    return BoundingBox2D(x_min, y_min, x_max - x_min, y_max - y_min)
