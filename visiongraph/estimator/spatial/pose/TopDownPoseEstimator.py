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
    def __init__(self,
                 human_detector: ObjectDetector[ObjectDetectionResult] = SSDDetector.create(
                     SSDConfig.PersonDetection_0200_256x256_FP32),
                 min_score: float = 0.5):
        super().__init__(min_score)

        self.human_detector = human_detector
        self.human_classes: Optional[Set[int]] = None

        # todo: use roi ratio for roi creation
        self.roi_ratio: Optional[float] = None
        self.roi_rectified = True

    def setup(self):
        self.human_detector.setup()

    def process(self, data: np.ndarray) -> ResultList[OutputType]:
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
        return image

    @abstractmethod
    def _detect_landmarks(self, image: np.ndarray, roi: np.ndarray, xs: int, ys: int) -> List[OutputType]:
        pass

    def release(self):
        self.human_detector.release()
