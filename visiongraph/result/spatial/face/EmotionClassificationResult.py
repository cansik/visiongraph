import cv2
import numpy as np

from visiongraph.result.ClassificationResult import ClassificationResult


class EmotionClassificationResult(ClassificationResult):
    def __init__(self, class_id: int, class_name: str, score: float, probabilities: np.ndarray):
        super().__init__(class_id, class_name, score)
        self.probabilities = probabilities

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        super().annotate(image, **kwargs)

        h, w = image.shape[:2]
        point = (int(x * 1.2 * w), int(y * 1.2 * h))
        cv2.putText(image, f"{self.class_name} {self.score:.2f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
