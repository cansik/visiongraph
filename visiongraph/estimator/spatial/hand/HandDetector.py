from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.HandDetectionResult import HandDetectionResult

OutputType = TypeVar('OutputType', bound=HandDetectionResult)


class FaceDetector(ObjectDetector[OutputType], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        pass
