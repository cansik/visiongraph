import cv2
import numpy as np

from visiongraph.result.ClassificationResult import ClassificationResult

"""
A class to represent an emotion classification result, inheriting from the base ClassificationResult.
"""


class EmotionClassificationResult(ClassificationResult):
    def __init__(self, class_id: int, class_name: str, score: float, probabilities: np.ndarray) -> None:
        """
        Initializes the EmotionClassificationResult object with the given parameters.

        Args:
            class_id (int): The ID of the classified class.
            class_name (str): The name of the classified class.
            score (float): The confidence score of the classification.
            probabilities (np.ndarray): An array of probabilities for each class in the classification result.
        """
        super().__init__(class_id, class_name, score)
        self.probabilities = probabilities

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs) -> None:
        """
        Annotates the given image with the classification result.

        Args:
            image (np.ndarray): The input image to be annotated.
            x (float): The x-coordinate of the annotation point. Defaults to 0.
            y (float): The y-coordinate of the annotation point. Defaults to 0.
            length (float): The length of the annotation box. Defaults to 0.2.

        Returns:
            None
        """
        super().annotate(image, **kwargs)

        h, w = image.shape[:2]
        point = (int(x * 1.2 * w), int(y * 1.2 * h))
        cv2.putText(image, f"{self.class_name} {self.score:.2f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
