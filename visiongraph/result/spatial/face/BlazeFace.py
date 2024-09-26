"""
BlazeFace class representing a face landmark result with additional attributes.
"""

import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.FaceLandmarkResult import FaceLandmarkResult


class BlazeFace(FaceLandmarkResult):
    """
    Initializes the BlazeFace object with score, landmarks and bounding box.
    
    Args:
        score (float): The confidence score of the detection.
        landmarks (vector.VectorNumpy4D): A 4-dimensional vector representing the 3D position and 1D prediction score.
        bounding_box (BoundingBox2D): A 2D bounding box containing the face detection results.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, bounding_box: BoundingBox2D):
        super().__init__(score, landmarks, bounding_box)

    @property
    """
    Returns the right eye as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the right eye.
    """
    def right_eye(self) -> vector.Vector4D:
        return self.landmarks[0]

    @property
    """
    Returns the left eye as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the left eye.
    """
    def left_eye(self) -> vector.Vector4D:
        return self.landmarks[1]

    @property
    """
    Returns the nose as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the nose.
    """
    def nose(self) -> vector.Vector4D:
        return self.landmarks[2]

    @property
    """
    Returns the mouth center as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the mouth center.
    """
    def mouth_center(self) -> vector.Vector4D:
        return self.landmarks[3]

    @property
    """
    Returns the right ear tragion as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the right ear tragion.
    """
    def right_ear_tragion(self) -> vector.Vector4D:
        return self.landmarks[4]

    @property
    """
    Returns the left ear tragion as a 3D position.
    
    Returns:
        vector.Vector4D: The 4-dimensional vector representing the x, y, z coordinates and prediction score of the left eartragion.
    """
    def left_ear_tragion(self) -> vector.Vector4D:
        return self.landmarks[5]
