from typing import TypeVar

import numpy as np

from visiongraph.result.EmbeddingResult import EmbeddingResult
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult

T = TypeVar("T", bound=LandmarkDetectionResult)


class LandmarkEmbeddingResult(EmbeddingResult):
    """
    A result class that wraps landmark detection results with embeddings.
    """

    def __init__(self, embeddings: np.ndarray, detection: T):
        """
        Initializes the LandmarkEmbeddingResult object.

        :param embeddings: The embedding data of detected landmarks.
        :param detection: The landemark detection result to be embedded.
        """
        super().__init__(embeddings)
        self.detection = detection

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the image with the detected landmarks.

        :param image: The input image to be annotated.
        :param **kwargs: Additional keyword arguments for the annotation process.
        """
        self.detection.annotate(image, **kwargs)
