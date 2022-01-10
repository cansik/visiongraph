from abc import ABC, abstractmethod

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ImageUtils import extract_roi_safe


class RoiEstimator(VisionEstimator[BaseResult], ABC):

    def process_roi(self, image: np.ndarray,
                    xmin: float, ymin: float, xmax: float, ymax: float, rectified: bool = True) -> BaseResult:
        roi, xs, ys = extract_roi_safe(image, xmin, ymin, xmax, ymax, rectified=rectified)
        result = self.process(roi)

        # used to transform result back to original image coordinates
        self._transform_result(result, image, roi, xs, ys)

        return result

    def process_detection(self, image: np.ndarray,
                          detection: ObjectDetectionResult, rectified: bool = True) -> BaseResult:
        bbox = detection.bounding_box
        return self.process_roi(image, bbox.x_min, bbox.y_min,
                                bbox.x_min + bbox.width, bbox.y_min + bbox.height, rectified)

    def _transform_result(self, result: BaseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        pass
