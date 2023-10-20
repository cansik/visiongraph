import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Optional, Union

import numpy as np

from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.result.ClassificationResult import ClassificationResult
from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.ResultList import ResultList

T = TypeVar("T", bound=EmbeddingResult)


class BaseKNNClassifier(BaseClassifier[ResultList[T], ResultList[ClassificationResult]], ABC):
    def __init__(self, min_score: float,
                 store_training_data: bool = True,
                 data_path: Optional[Union[str, os.PathLike]] = None):
        super().__init__(min_score)

        self.store_training_data = store_training_data

        self._training_data: Optional[np.ndarray] = None
        self._data_labels = np.array([], dtype=int)

        self._data_path = data_path

    @abstractmethod
    def setup(self):
        if self._data_path is not None:
            self.load_data(self._data_path)

    def add_sample(self, embedding_result: T, label_index: int):
        self.add_samples(np.array([embedding_result.embeddings]), np.array([label_index]))

    @abstractmethod
    def add_samples(self, x: np.ndarray, y: np.ndarray):
        self._data_labels = np.append(self._data_labels, y.astype(int))

        if self.store_training_data:
            if self._training_data is None:
                self._training_data = x
            else:
                self._training_data = np.vstack((self._training_data, x))

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

    def save_data(self, path: Union[str, os.PathLike]):
        if self.training_data is None:
            logging.warning("Training data is empty!")
            return

        path = Path(path)
        np.savez_compressed(path, x=self._training_data, y=self._data_labels, labels=self.labels)

    def load_data(self, path: Union[str, os.PathLike]):
        path = Path(path)
        data = np.load(path)

        x = data["x"]
        y = data["y"]
        labels = data["labels"]

        self.labels = labels.tolist()
        self.add_samples(x, y)

    @property
    def training_data(self) -> Optional[np.ndarray]:
        return self._training_data

    @property
    def training_data_labels(self) -> np.ndarray:
        return self._data_labels
