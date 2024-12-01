from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.result.CameraPoseResult import CameraPoseResult


class BoardCameraCalibrator(VisionEstimator[Optional[CameraPoseResult]], ABC):
    """
    Abstract base class for calibrating board cameras.

    This class provides a basic interface for calibrating board cameras, allowing for customization and extension.
    """

    def __init__(self, columns: int, rows: int, max_samples: int = -1):
        """
        Initializes the BoardCameraCalibrator with the number of columns and rows on the board,
        as well as an optional maximum number of samples to collect.

        :param columns: The number of columns on the board.
        :param rows: The number of rows on the board.
        :param max_samples: The maximum number of samples to collect. Defaults to -1, which means no limit.
        """
        self.max_samples = max_samples

        self.rows = rows
        self.columns = columns

        self.board_detected: bool = False

    def setup(self):
        """
        Sets up the calibrator for use.

        This method should be implemented by concrete classes to prepare the necessary resources and settings.
        """
        pass

    @abstractmethod
    def process(self, data: np.ndarray) -> Optional[CameraPoseResult]:
        """
        Processes a batch of image data from the board camera.

        :param data: The input image data.

        :return: The result of the processing, or None if no result is available.
        """
        pass

    @abstractmethod
    def calibrate(self) -> Optional[CameraPoseResult]:
        """
        Calibrates the board camera.

        This method should be implemented by concrete classes to perform the actual calibration process,
        which may involve collecting multiple samples and computing the pose of each sample.

        :return: The calibrated pose result, or None if no result is available.
        """
        pass

    def release(self):
        """
        Releases any resources held by the calibrator.
        """
        pass

    @property
    @abstractmethod
    def sample_count(self):
        """
        Gets the total number of samples collected.

        :return: The total number of samples collected, or -1 if no samples have been collected.
        """
        pass
