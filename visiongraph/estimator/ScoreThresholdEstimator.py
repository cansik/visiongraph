from abc import ABC
from typing import TypeVar

from visiongraph.estimator.BaseEstimator import BaseEstimator
from visiongraph.result.BaseResult import BaseResult

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType', bound=BaseResult)


class ScoreThresholdEstimator(BaseEstimator[InputType, OutputType], ABC):
    """
    An estimator that determines the threshold score for a model based on a minimum required score.
    """

    def __init__(self, min_score: float) -> None:
        """
        Initializes the ScoreThresholdEstimator with a minimum score threshold.

        :param min_score: The minimum score required to pass through the estimator.
        """
        self.min_score = min_score
