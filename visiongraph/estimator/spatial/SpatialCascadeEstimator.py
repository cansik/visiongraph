from typing import List, Dict

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.spatial.SpatialCascadeResult import SpatialCascadeResult


class SpatialCascadeEstimator(ObjectDetector):
    def __init__(self, root_detector: ObjectDetector, **child_detectors: RoiEstimator):
        super().__init__(min_score=0)
        self.root_detector = root_detector
        self.child_detectors: Dict[str, RoiEstimator] = child_detectors

        self._detectors = [self.root_detector, *self.child_detectors.values()]

    def setup(self):
        for detector in self._detectors:
            detector.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> List[SpatialCascadeResult]:
        root_results = self.root_detector.estimate(image, **kwargs)

        results = []
        for root_result in root_results:
            child_results = {}

            for name, detector in self.child_detectors.items():
                result = detector.estimate_detection(image, root_result, **kwargs)
                child_results.update({name: result})

            results.append(SpatialCascadeResult(root_result, **child_results))

        return results

    def release(self):
        for detector in self._detectors:
            detector.release()
