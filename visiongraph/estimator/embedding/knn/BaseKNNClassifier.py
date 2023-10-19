from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.result.ClassificationResult import ClassificationResult
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.ResultList import ResultList

T = TypeVar("T", bound=EmbeddingResult)


class BaseKNNClassifier(BaseClassifier[ResultList[T], ResultList[ClassificationResult]], ABC):

    def add_sample(self, embedding_result: T, label_index: int):
        self.add_samples(np.array([embedding_result.embeddings]), np.array([label_index]))

    @abstractmethod
    def add_samples(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, embedding_result: T) -> ClassificationResult:
        results = self.predict_all(np.array([embedding_result.embeddings]))
        predicted_index = int(results[0][0])
        score = float(results[0][1])
        return ClassificationResult(predicted_index, self.get_label(predicted_index), score)

    @abstractmethod
    def predict_all(self, x: np.ndarray) -> np.ndarray:
        """
        Returns np.ndarray of shape (n, 2) which contains class indexes and scores.
        """
        pass

    def process(self, embedding_results: ResultList[T]) -> ResultList[ClassificationResult]:
        results = self.predict_all(np.array(
            [r.embeddings for r in embedding_results]
        ))

        classifications = ResultList()
        for result in results:
            predicted_index = int(result[0])
            score = float(result[1])
            classifications.append(ClassificationResult(predicted_index, self.get_label(predicted_index), score))

        return classifications
