from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

HAND_DETECTION_ID = 0
HAND_DETECTION_LABEL = "hand"


class HandDetectionResult(ObjectDetectionResult):
    def __init__(self, score: float, bounding_box: BoundingBox2D):
        super().__init__(HAND_DETECTION_ID, HAND_DETECTION_LABEL, score, bounding_box)
