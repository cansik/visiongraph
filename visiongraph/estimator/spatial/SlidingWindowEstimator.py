from argparse import Namespace
from typing import TypeVar, Tuple, List

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util import ResultUtils

OutputType = TypeVar('OutputType', bound=ObjectDetectionResult)
"""
Generic type variable for the output type of the Object Detection result derived from ObjectDetectionResult.
"""


class SlidingWindowEstimator(ObjectDetector[OutputType]):
    """
    Sliding Window Estimator that applies object detection on sliding windows.
    """

    def __init__(self, network: ObjectDetector[OutputType],
                 step_size: int, window_size: Tuple[int, int],
                 min_score: float = 0.5, iou_threshold: float = 0.3):
        """
        Initializes the Sliding Window Estimator.

        :param network: The object detection network.
        :param step_size: The step size for sliding the window.
        :param window_size: The size of the sliding window.
        :param min_score: The minimum score threshold for detections (default is 0.5).
        :param iou_threshold: The Intersection over Union threshold for non-maximum suppression (default is 0.3).
        """
        super().__init__(min_score)

        self.network = network
        self.step_size = step_size
        self.window_size = window_size
        self.iou_threshold = iou_threshold

    def setup(self):
        """
        Setup the internal network for processing.
        """
        self.network.setup()

    def process(self, frame: np.ndarray) -> ResultList[OutputType]:
        """
        Process the input frame using sliding window object detection.

        :param frame: The input frame for object detection.

        :return: A list of detected object results.
        """
        detections: List[OutputType] = []
        ih, iw = frame.shape[:2]

        for x, y, roi in self._sliding_window(frame, self.step_size, self.window_size):
            results = self.network.process(roi)

            h, w = roi.shape[:2]
            for result in results:
                result.map_coordinates(self.window_size, (iw, ih), dest_roi=BoundingBox2D(x, y, w, h))
                detections.append(result)

        # perform nms on detections
        final_detections = ResultUtils.non_maximum_suppression(detections, self.min_score, self.iou_threshold)

        return ResultList(final_detections)

    def release(self):
        """
        Release internal resources.
        """
        self.network.release()

    def configure(self, args: Namespace):
        """
        Configure the internal network with the provided arguments.

        :param args: The namespace containing configuration arguments.
        """
        self.network.configure(args)

    @staticmethod
    def _sliding_window(image, step_size, window_size) -> Tuple[int, int, np.ndarray]:
        """
        Generate sliding windows over an input image.

        :param image: The input image.
        :param step_size: The step size for sliding the window.
        :param window_size: The size of the sliding window.

:param Yields: 
        """
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                yield x, y, image[y:y + window_size[1], x:x + window_size[0]]
