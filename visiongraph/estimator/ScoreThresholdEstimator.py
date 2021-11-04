from abc import ABC

from visiongraph.estimator.BaseEstimator import BaseEstimator


class ScoreThresholdEstimator(BaseEstimator, ABC):
    def __init__(self, min_score: float):
        self.min_score = min_score
