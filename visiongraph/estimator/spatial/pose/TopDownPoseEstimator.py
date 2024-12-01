from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Set, List

import numpy as np

from visiongraph.estimator.spatial.LandmarkEstimator import LandmarkEstimator
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.SSDDetector import SSDDetector, SSDConfig
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util import ImageUtils

OutputType = TypeVar('OutputType', bound=PoseLandmarkResult)


class TopDownPoseEstimator(LandmarkEstimator[OutputType], ABC):
    """
    A top-down pose estimator that uses a human detector to detect people and then estimates their pose.
    """

    def __init__(self,
                 human_detector: ObjectDetector[ObjectDetectionResult] = SSDDetector.create(
                     SSDConfig.PersonDetection_0200_256x256_FP32),
                 min_score: float = 0.5):
        """
        Initializes the TopDownPoseEstimator.

        :param human_detector: The human detector used to detect people.
        :param min_score: The minimum score required for a detection to be considered valid.
        """
        super().__init__(min_score)

        self.human_detector = human_detector
        self.human_classes: Optional[Set[int]] = None

        # todo: use roi ratio for roi creation
        self.roi_ratio: Optional[float] = None
        self.roi_rectified = True

    def setup(self):
        """
        Sets up the human detector.
        """
        self.human_detector.setup()

    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        """
        Processes an image to detect people and estimate their pose.

        :param data: The input image.

        :return: A list of pose landmark results.
        """
        detections: List[ObjectDetectionResult] = self.human_detector.process(data)

        # filter non-human classes
        if self.human_classes is not None:
            detections = [d for d in detections if d.class_id in self.human_classes]

        data = self._pre_landmark(data)

        results: ResultList[OutputType] = ResultList()
        for detection in detections:
            # extract roi
            xmin, ymin, xmax, ymax = detection.bounding_box.to_array(tl_br_format=True)
            roi, xs, ys = ImageUtils.extract_roi_safe(data, xmin, ymin, xmax, ymax, rectified=self.roi_rectified)

            poses = self._detect_landmarks(data, roi, xs, ys)
            results += poses

        return results

    def _pre_landmark(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-processes an image by doing nothing in this implementation.

        :param image: The input image.

        :return: The pre-processed image.
        """
        return image

    @abstractmethod
    def _detect_landmarks(self, image: np.ndarray, roi: np.ndarray, xs: int, ys: int) -> List[OutputType]:
        """
        Detects landmarks in a region of interest (ROI).

        :param image: The input image.
        :param roi: The region of interest.
        :param xs: The x-coordinate of the ROI center.
        :param ys: The y-coordinate of the ROI center.

        :return: A list of pose landmark results.
        """
        pass

    def release(self):
        """
        Releases any resources used by the estimator.
        """
        self.human_detector.release()
