from abc import ABC, abstractmethod
from typing import List, TypeVar, Sequence, Union

from visiongraph.estimator.ScoreThresholdEstimator import ScoreThresholdEstimator
from visiongraph.result.ClassificationResult import ClassificationResult

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=ClassificationResult)


class BaseClassifier(ScoreThresholdEstimator[InputType, OutputType], ABC):

    def __init__(self, min_score: float):
        super().__init__(min_score)
        self.labels: List[str] = []

    @abstractmethod
    def process(self, data: InputType) -> OutputType:
        pass

    def get_label(self, index: int) -> str:
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return str(index)
