from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.CameraPoseResult import CameraPoseResult


class BoardCameraCalibrator(VisionEstimator[Optional[CameraPoseResult]], ABC):
    def __init__(self, columns: int, rows: int, max_samples: int = -1):
        self.max_samples = max_samples

        self.rows = rows
        self.columns = columns

        self.board_detected: bool = False

    def setup(self):
        pass

    @abstractmethod
    def process(self, data: np.ndarray) -> Optional[CameraPoseResult]:
        pass

    @abstractmethod
    def calibrate(self) -> Optional[CameraPoseResult]:
        pass

    def release(self):
        pass

    @property
    @abstractmethod
    def sample_count(self):
        pass
