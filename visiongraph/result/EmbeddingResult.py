import numpy as np

from visiongraph.result.BaseResult import BaseResult
from scipy.spatial.distance import cosine


class EmbeddingResult(BaseResult):
    def __init__(self, embeddings: np.ndarray, landmark_overlap: float):
        self.embeddings = embeddings
        self.landmark_overlap = landmark_overlap

    def annotate(self, image: np.ndarray, **kwargs):
        pass

    def cosine_dist(self, embeddings: np.ndarray):
        return cosine(self.embeddings, embeddings) * 0.5
