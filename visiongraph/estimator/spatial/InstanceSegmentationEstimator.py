from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult

OutputType = TypeVar('OutputType', bound=InstanceSegmentationResult)


class InstanceSegmentationEstimator(ObjectDetector[OutputType], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        pass
