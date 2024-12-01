import numpy as np

from visiongraph.result.BaseResult import BaseResult


class ClassificationResult(BaseResult):
    """
    A class to represent a classification result with class ID, name, and score.
    """

    def __init__(self, class_id: int, class_name: str, score: float):
        """
        Initializes the ClassificationResult object.

        :param class_id: The ID of the classified class.
        :param class_name: The name of the classified class.
        :param score: The confidence score of the classification result.
        """
        self.class_id = class_id
        self.class_name = class_name
        self.score = score

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Adds an annotation to the given image.

        :param image: The input image.
        :param **kwargs: Additional keyword arguments to be passed to the super class's annotate method.
        """
        super().annotate(image, **kwargs)
