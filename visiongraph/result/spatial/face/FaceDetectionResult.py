from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

"""
Face Detection Result class.

This class extends the ObjectDetectionResult to provide a specific result for face detection tasks.
"""

FACE_DETECTION_ID = 0
FACE_DETECTION_LABEL = "face"


class FaceDetectionResult(ObjectDetectionResult):
    """
    Represents the result of a face detection task.
    """

    def __init__(self, score: float, bounding_box: BoundingBox2D):
        """
        Initializes the FaceDetectionResult object with a given score and bounding box.

        :param score: The confidence score of the detected face.
        :param bounding_box: The bounding box of the detected face.
        """
        super().__init__(FACE_DETECTION_ID, FACE_DETECTION_LABEL, score, bounding_box)
