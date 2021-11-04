from abc import ABC, abstractmethod
from typing import List

from visiongraph.estimator.ScoreThresholdEstimator import ScoreThresholdEstimator
from visiongraph.result.ClassificationResult import ClassificationResult


class BaseClassifier(ScoreThresholdEstimator, ABC):

    def __init__(self, min_score: float):
        super().__init__(min_score)
        self.labels: List[str] = []

    @abstractmethod
    def estimate(self, **kwargs) -> ClassificationResult:
        pass
