from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

HAND_DETECTION_ID = 0
HAND_DETECTION_LABEL = "hand"


class HandDetectionResult(ObjectDetectionResult):
    """
    Represents the result of a hand detection task.
    """

    def __init__(self, score: float, bounding_box: BoundingBox2D) -> None:
        """
        Initializes the HandDetectionResult object.

        :param score: The confidence score of the detection.
        :param bounding_box: The bounding box of the detected object.
        """
        super().__init__(HAND_DETECTION_ID, HAND_DETECTION_LABEL, score, bounding_box)
