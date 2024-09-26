import numpy as np

from visiongraph.result.BaseResult import BaseResult


class ClassificationResult(BaseResult):
    """
    A class to represent a classification result with class ID, name, and score.
    """

    def __init__(self, class_id: int, class_name: str, score: float):
        """
        Initializes the ClassificationResult object.

        Args:
            class_id (int): The ID of the classified class.
            class_name (str): The name of the classified class.
            score (float): The confidence score of the classification result.
        """
        self.class_id = class_id
        self.class_name = class_name
        self.score = score

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Adds an annotation to the given image.

        Args:
            image (np.ndarray): The input image.
            **kwargs: Additional keyword arguments to be passed to the super class's annotate method.
        """
        super().annotate(image, **kwargs)
