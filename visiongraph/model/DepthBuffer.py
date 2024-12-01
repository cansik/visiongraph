from abc import ABC, abstractmethod
from statistics import median
from typing import List, Tuple

import numpy as np


class DepthBuffer(ABC):
    """
    Abstract base class for a depth buffer.
    This class defines the interface for calculating distances
    from a point to a depth buffer and retrieving depth buffer data.
    """

    @abstractmethod
    def distance(self, x: float, y: float) -> float:
        """
        Calculates the distance from a point to the depth buffer.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.

        :return: The calculated distance.
        """
        pass

    def median_distance(self, points: List[Tuple[float, float]]) -> float:
        """
        Calculates the median distance from a list of points to the depth buffer.

        :param points: A list of 2D points.

        :return: The median calculated distance.
        """
        return median([self.distance(p[0], p[1]) for p in points])

    @property
    @abstractmethod
    def depth_buffer(self) -> np.ndarray:
        """
        Abstract property to retrieve the depth buffer data.

        :return: The underlying depth buffer as a NumPy array.
        """
        pass

    @property
    @abstractmethod
    def depth_map(self) -> np.ndarray:
        """
        Abstract property to retrieve the depth map.

        :return: The depth map as a NumPy array.
        """
        pass
