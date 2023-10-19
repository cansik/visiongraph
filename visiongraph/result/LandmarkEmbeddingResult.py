from typing import TypeVar

import numpy as np

from visiongraph.result.BaseResult import BaseResult
from scipy.spatial.distance import cosine

from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

T = TypeVar("T", bound=LandmarkDetectionResult)


class LandmarkEmbeddingResult(BaseResult):
    def __init__(self, embeddings: np.ndarray, detection: T):
        self.embeddings = embeddings
        self.detection = detection

    def annotate(self, image: np.ndarray, **kwargs):
        pass

    def cosine_dist(self, embeddings: np.ndarray):
        return cosine(self.embeddings, embeddings) * 0.5
