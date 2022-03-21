import numpy as np

from visiongraph.result.BaseResult import BaseResult


class EmbeddingResult(BaseResult):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def annotate(self, image: np.ndarray, **kwargs):
        pass
