from abc import ABC

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ImageUtils import extract_roi_safe


class RoiEstimator(VisionEstimator[BaseResult], ABC):
    """
    A class for estimating regions of interest (ROI) in images.

    This class provides methods to process ROI or object detection results in images.
    """

    def process_roi(self, image: np.ndarray,
                    xmin: float, ymin: float, xmax: float, ymax: float, rectified: bool = True) -> BaseResult:
        """
        Processes a region of interest (ROI) in an image.

        Args:
            image (np.ndarray): The input image.
            xmin (float): The minimum x-coordinate of the ROI.
            ymin (float): The minimum y-coordinate of the ROI.
            xmax (float): The maximum x-coordinate of the ROI.
            ymax (float): The maximum y-coordinate of the ROI.
            rectified (bool, optional): A flag to specify if the ROI is rectified. Defaults to True.

        Returns:
            BaseResult: The result of processing the ROI.
        """
        roi, xs, ys = extract_roi_safe(image, xmin, ymin, xmax, ymax, rectified=rectified)
        result = self.process(roi)

        # used to transform result back to original image coordinates
        self._transform_result(result, image, roi, xs, ys)

        return result

    def process_detection(self, image: np.ndarray,
                          detection: ObjectDetectionResult, rectified: bool = True) -> BaseResult:
        """
        Processes an object detection result within the image.

        Args:
            image (np.ndarray): The input image.
            detection (ObjectDetectionResult): The object detection result.
            rectified (bool, optional): A flag to specify if the ROI is rectified. Defaults to True.

        Returns:
            BaseResult: The result of processing the object detection result.
        """
        bbox = detection.bounding_box
        return self.process_roi(image, bbox.x_min, bbox.y_min,
                                bbox.x_min + bbox.width, bbox.y_min + bbox.height, rectified)

    def _transform_result(self, result: BaseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        """
        Transforms a result back to the original image coordinates.

        Args:
            result (BaseResult): The result to be transformed.
            image (np.ndarray): The original image.
            roi (np.ndarray): The region of interest in the image.
            xs (float): The x coordinate scale factor.
            ys (float): The y coordinate scale factor.
        """
        pass
