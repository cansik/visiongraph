from abc import ABC, abstractmethod
from typing import List, Tuple
from statistics import median

import numpy as np


class DepthBuffer(ABC):
    @abstractmethod
    def distance(self, x: float, y: float) -> float:
        pass

    def median_distance(self, points: List[Tuple[float, float]]) -> float:
        return median([self.distance(p[0], p[1]) for p in points])

    @property
    @abstractmethod
    def depth_buffer(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def depth_map(self) -> np.ndarray:
        pass
