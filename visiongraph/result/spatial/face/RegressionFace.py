from typing import Optional

import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class RegressionFace(FaceLandmarkResult):
    """
    Represents a face regression result, containing facial landmarks and bounding box information.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D,
                 bounding_box: Optional[BoundingBox2D] = None):
        """
        Initializes the RegressionFace object.

        Args:
            score (float): The face detection confidence score.
            landmarks (vector.VectorNumpy4D): A vector of facial landmarks as Numpy arrays.
            bounding_box (BoundingBox2D, optional): The bounding box enclosing the face. Defaults to None.
        """
        super().__init__(score, landmarks, bounding_box)

    @property
    def left_eye(self) -> vector.Vector4D:
        """
        Gets the 3D position of the left eye.

        Returns:
            vector(Vector4D): The 3D position of the left eye as a Vector4D.
        """
        return self.landmarks[0]

    @property
    def right_eye(self) -> vector.Vector4D:
        """
        Gets the 3D position of the right eye.

        Returns:
            vector(Vector4D): The 3D position of the right eye as a Vector4D.
        """
        return self.landmarks[1]

    @property
    def nose(self) -> vector.Vector4D:
        """
        Gets the 3D position of the nose tip.

        Returns:
            vector(Vector4D): The 3D position of the nose tip as a Vector4D.
        """
        return self.landmarks[2]

    @property
    def mouth_left(self) -> vector.Vector4D:
        """
        Gets the 2D position of the left mouth corner.

        Returns:
            vector(Vector4D): The 2D position of the left mouth corner as a Vector4D.
        """
        return self.landmarks[3]

    @property
    def mouth_right(self) -> vector.Vector4D:
        """
        Gets the 2D position of the right mouth corner.

        Returns:
            vector(Vector4D): The 2D position of the right mouth corner as a Vector4D.
        """
        return self.landmarks[4]
