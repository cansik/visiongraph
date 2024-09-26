import numpy as np
from scipy.spatial.distance import cosine

from visiongraph.result.BaseResult import BaseResult


class EmbeddingResult(BaseResult):
    """
    A class to represent the result of an embedding computation.
    """

    def __init__(self, embeddings: np.ndarray):
        """
        Initializes the EmbeddingResult object.

        Args:
            embeddings (np.ndarray): The embedded vectors.
        """
        self.embeddings = embeddings

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the result with additional information.

        Args:
            image (np.ndarray): The input image.
        """
        pass

    def cosine_dist(self, embeddings: np.ndarray) -> float:
        """
        Computes the cosine distance between this embedding and another.

        Args:
            embeddings (np.ndarray): The other embedded vectors.

        Returns:
            float: The cosine distance between this embedding and the input.
        """
        return cosine(self.embeddings, embeddings) * 0.5
