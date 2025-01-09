import cv2
import numpy as np

from visiongraph.result.ClassificationResult import ClassificationResult

SPEAKING_LABEL = "speaking"
NON_SPEAKING_LABEL = "non-speaking"


class VivaVADResult(ClassificationResult):
    """
    Represents the result of a voice activity detection (VAD) classification.
    """

    def __init__(self, class_id: int, class_name: str, score: float, probabilities: np.ndarray):
        """
        Initializes a VivaVADResult instance.

        :param class_id: The ID of the predicted class.
        :param class_name: The name of the predicted class.
        :param score: The confidence score of the prediction.
        :param probabilities: The probability distribution over all classes.
        """
        super().__init__(class_id, class_name, score)
        self.probabilities = probabilities

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, spacing: float = 0.01, **kwargs):
        """
        Annotates the given image with the VAD result.

        :param image: The image to annotate.
        :param x: The horizontal position of the annotation, normalized (0 to 1).
        :param y: The vertical position of the annotation, normalized (0 to 1).
        :param spacing: The spacing factor for positioning the annotation.
        :param kwargs: Additional keyword arguments for annotation customization.
        """
        super().annotate(image, **kwargs)

        h, w = image.shape[:2]
        spacing = spacing + 1
        point = (int(x * spacing * w), int(y * spacing * h))
        cv2.putText(image, f"{self.class_name} {self.score:.2f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

    @property
    def is_speaking(self) -> bool:
        """
        Indicates whether the result corresponds to a speaking activity.

        :return: True if the predicted class is "speaking"; otherwise, False.
        """
        return self.class_id == 1

    @property
    def speaking_score(self) -> float:
        """
        Retrieves the confidence score for the "speaking" class.

        :return: The confidence score if the prediction is "speaking";
                 otherwise, the inverse confidence score.
        """
        if self.is_speaking:
            return self.score
        return 1.0 - self.score
