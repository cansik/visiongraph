from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.VisionClassifier import VisionClassifier
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

OutputType = TypeVar('OutputType', bound=ObjectDetectionResult)


class ObjectDetector(VisionClassifier[ResultList[OutputType]], ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> ResultList[OutputType]:
        pass
