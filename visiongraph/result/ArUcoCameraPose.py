import numpy as np
import vector

from visiongraph.result.ArUcoMarkerDetection import ArUcoMarkerDetection
from visiongraph.result.BaseResult import BaseResult


class ArUcoCameraPose(BaseResult):
    """
    Represents the camera pose with respect to an ArUco marker.
    """

    def __init__(self, position: vector.Vector3D, rotation: vector.Vector3D, marker: ArUcoMarkerDetection):
        """
        Initializes an instance of ArUcoCameraPose.

        Args:
            position (vector.Vector3D): The 3D position of the camera.
            rotation (vector.Vector3D): The 3D orientation of the camera.
            marker (ArUcoMarkerDetection): The detected ArUco marker.
        """
        self.position = position
        self.rotation = rotation
        self.marker: ArUcoMarkerDetection = marker

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the input image with additional information.

        Args:
            image (np.ndarray): The input image to be annotated.
            **kwargs: Additional keyword arguments to pass to the base result's annotation method.

        Returns:
            None
        """
        super().annotate(image, **kwargs)
        self.marker.annotate(image, **kwargs)
