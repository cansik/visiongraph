from abc import abstractmethod

import numpy as np

from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult


class HeadPoseEstimator(RoiEstimator):
    """
    An abstract base class for head pose estimators.

    It extends the RoiEstimator class and provides a common interface for head pose estimation.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> HeadPoseResult:
        """
        Processes an input image to estimate the head pose.

        Args:
            image (np.ndarray): The input image to be processed.

        Returns:
            HeadPoseResult: The estimated head pose result.
        """
        pass
