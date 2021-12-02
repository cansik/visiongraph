from typing import List

import numpy as np

from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.spatial.pose.BlazePose import BlazePose


class MediaPipePoseEstimator(PoseEstimator):

    def setup(self):
        pass

    def estimate(self, image: np.ndarray, **kwargs) -> List[BlazePose]:
        pass

    def release(self):
        pass