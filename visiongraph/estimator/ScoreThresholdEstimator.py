from abc import ABC
from typing import TypeVar

from visiongraph.result.BaseResult import BaseResult
from visiongraph.estimator.BaseEstimator import BaseEstimator

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=BaseResult)


class ScoreThresholdEstimator(BaseEstimator[InputType, OutputType], ABC):
    def __init__(self, min_score: float):
        self.min_score = min_score
