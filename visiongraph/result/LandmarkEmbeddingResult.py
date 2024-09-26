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

        Args:
            embeddings (np.ndarray): The embedding data of detected landmarks.
            detection (T): The landemark detection result to be embedded.
        """
        super().__init__(embeddings)
        self.detection = detection

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the image with the detected landmarks.

        Args:
            image (np.ndarray): The input image to be annotated.
            **kwargs: Additional keyword arguments for the annotation process.
        """
        self.detection.annotate(image, **kwargs)
