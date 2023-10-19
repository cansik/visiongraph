from typing import TypeVar

import numpy as np

from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

T = TypeVar("T", bound=LandmarkDetectionResult)


class LandmarkEmbeddingResult(EmbeddingResult):
    def __init__(self, embeddings: np.ndarray, detection: T):
        super().__init__(embeddings)
        self.detection = detection

    def annotate(self, image: np.ndarray, **kwargs):
        self.detection.annotate(image, **kwargs)
