from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceDetectionResult import FaceDetectionResult

OutputType = TypeVar('OutputType', bound=FaceDetectionResult)


class FaceDetector(ObjectDetector[OutputType], ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> ResultList[OutputType]:
        pass
