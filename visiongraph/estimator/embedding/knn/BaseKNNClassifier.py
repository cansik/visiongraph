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
        """
        Initializes the K-Nearest Neighbors classifier.

        :param min_score: Minimum score for classification.
        :param store_training_data: Flag to store training data. Defaults to True.
        :param data_path: Path to load data from. Defaults to None.
        """
        super().__init__(min_score)

        self.store_training_data = store_training_data

        self._training_data: Optional[np.ndarray] = None
        self._data_labels = np.array([], dtype=int)

        self._data_path = data_path

    @abstractmethod
    def setup(self):
        """
        Sets up the classifier by loading the training data.
        """
        if self._data_path is not None:
            self.load_data(self._data_path)

    def add_sample(self, embedding_result: T, label_index: int):
        """
        Adds a single sample to the classifier.

        :param embedding_result: Embedding result of the sample.
        :param label_index: Index of the corresponding label.
        """
        self.add_samples(np.array([embedding_result.embeddings]), np.array([label_index]))

    @abstractmethod
    def add_samples(self, x: np.ndarray, y: np.ndarray):
        """
        Adds multiple samples to the classifier.

        :param x: Embeddings of the samples.
        :param y: Corresponding labels of the samples.
        """
        self._data_labels = np.append(self._data_labels, y.astype(int))

        if self.store_training_data:
            if self._training_data is None:
                self._training_data = x
            else:
                self._training_data = np.vstack((self._training_data, x))

    def predict(self, embedding_result: T) -> ClassificationResult:
        """
        Predicts the class of a single sample.

        :param embedding_result: Embedding result of the sample.

        :return: Classification result containing predicted index and score.
        """
        results = self.predict_all(np.array([embedding_result.embeddings]))
        predicted_index = int(results[0][0])
        score = float(results[0][1])
        return ClassificationResult(predicted_index, self.get_label(predicted_index), score)

    @abstractmethod
    def predict_all(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the classes of multiple samples.

        :param x: Embeddings of the samples.

        :return: Array containing predicted indexes and scores (n, 2).
        """
        pass

    def process(self, embedding_results: ResultList[T]) -> ResultList[ClassificationResult]:
        """
        Processes a list of embedding results and returns classification results.

        :param embedding_results: List of embedding results.

        :return: List of classification results.
        """
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
        """
        Saves the training data to a file.

        :param path: Path to save the data.
        """
        if self.training_data is None:
            logging.warning("Training data is empty!")
            return

        path = Path(path)
        np.savez_compressed(path, x=self._training_data, y=self._data_labels, labels=self.labels)

    def load_data(self, path: Union[str, os.PathLike]):
        """
        Loads the training data from a file.

        :param path: Path to load the data.
        """
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
        """
        Gets the training data labels.

        :return: Training data labels.
        """
        return self._data_labels
