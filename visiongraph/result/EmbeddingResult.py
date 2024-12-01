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

        :param embeddings: The embedded vectors.
        """
        self.embeddings = embeddings

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the result with additional information.

        :param image: The input image.
        """
        pass

    def cosine_dist(self, embeddings: np.ndarray) -> float:
        """
        Computes the cosine distance between this embedding and another.

        :param embeddings: The other embedded vectors.

        :return: The cosine distance between this embedding and the input.
        """
        return cosine(self.embeddings, embeddings) * 0.5
