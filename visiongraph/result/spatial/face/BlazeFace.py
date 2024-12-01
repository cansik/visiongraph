import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class BlazeFace(FaceLandmarkResult):
    """
    A class to represent facial landmarks detected by the BlazeFace model.

    Inherits from:
        FaceLandmarkResult: A result class that provides common functionalities for face landmark results.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: BoundingBox2D):
        """
        Initializes a BlazeFace object with the provided score, landmarks, and bounding box.

        :param score: The confidence score of the detected face.
        :param landmarks: The 2D landmarks of the face.
        :param bounding_box: The bounding box around the detected face.
        """
        super().__init__(score, landmarks, bounding_box)

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the right eye landmark.

        :return: The 2D coordinates of the right eye.
        """
        return self.landmarks[0]

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the left eye landmark.

        :return: The 2D coordinates of the left eye.
        """
        return self.landmarks[1]

    @property
    def nose(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the nose landmark.

        :return: The 2D coordinates of the nose.
        """
        return self.landmarks[2]

    @property
    def mouth_center(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the mouth center landmark.

        :return: The 2D coordinates of the mouth center.
        """
        return self.landmarks[3]

    @property
    def right_ear_tragion(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the right ear tragion landmark.

        :return: The 2D coordinates of the right ear tragion.
        """
        return self.landmarks[4]

    @property
    def left_ear_tragion(self) -> vector.Vector4D:
        """
        Retrieves the coordinates of the left ear tragion landmark.

        :return: The 2D coordinates of the left ear tragion.
        """
        return self.landmarks[5]
