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

        :param position: The 3D position of the camera.
        :param rotation: The 3D orientation of the camera.
        :param marker: The detected ArUco marker.
        """
        self.position = position
        self.rotation = rotation
        self.marker: ArUcoMarkerDetection = marker

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the input image with additional information.

        :param image: The input image to be annotated.
        :param **kwargs: Additional keyword arguments to pass to the base result's annotation method.

        """
        super().annotate(image, **kwargs)
        self.marker.annotate(image, **kwargs)
