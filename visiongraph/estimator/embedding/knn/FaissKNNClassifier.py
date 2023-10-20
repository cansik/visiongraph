import os
from argparse import ArgumentParser, Namespace
from typing import Optional, Union

import numpy as np
from faiss import IndexFlat, IndexFlatL2

from visiongraph.estimator.embedding.knn.BaseKNNClassifier import BaseKNNClassifier


class FaissKNNClassifier(BaseKNNClassifier):

    def __init__(self, index_dimensions: Optional[int] = None,
                 store_training_data: bool = True,
                 data_path: Optional[Union[str, os.PathLike]] = None):
        super().__init__(min_score=0.5,
                         store_training_data=store_training_data,
                         data_path=data_path)
        self.index: Optional[IndexFlat] = None
        self.index_dimensions = index_dimensions

    def setup(self):
        if self.index_dimensions is not None and self.index is not None:
            self.reset_index(self.index_dimensions)
        super().setup()

    def add_samples(self, x: np.ndarray, y: np.ndarray):
        super().add_samples(x, y)

        #  lazy init index
        if self.index is None:
            self.reset_index(x.shape[1])

        self.index.add(x.astype(np.float32))

    def predict_all(self, x: np.ndarray) -> np.ndarray:
        #  lazy init index
        if self.index is None:
            self.reset_index(x.shape[1])

        if len(self.labels) == 0 or self._data_labels.shape[0] == 0 or x.shape[0] == 0:
            return np.full((x.shape[0], 2), -1)

        # predict
        distances, indices = self.index.search(x.astype(np.float32), k=len(self.labels))

        best_indices = indices[:, 0].reshape(-1, 1)
        best_distances = distances[:, 0].reshape(-1, 1)

        # lookup class for index
        classes = self._data_labels[best_indices]

        return np.hstack((classes, best_distances))

    def reset_index(self, index_dimensions: int):
        if self.index is None:
            self.index = IndexFlatL2(index_dimensions)
        else:
            self.index.reset()
            self._data_labels = np.array([], dtype=int)

    def release(self):
        pass

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
