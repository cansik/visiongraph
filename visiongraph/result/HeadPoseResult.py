import numpy as np
import vector

from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.DrawingUtils import draw_axis


class HeadPoseResult(BaseResult):
    """
    A class to represent the result of head pose estimation, containing rotation and drawing utilities.
    """

    def __init__(self, rotation: vector.Vector3D):
        """
        Initializes the HeadPoseResult object with a specified rotation.

        :param rotation: The 3D rotation of the head.
        """
        self.rotation = rotation

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        """
        Draws a bounding box around the detected face on the provided image.

        :param image: The input image.
        :param x: The x-coordinate of the center of the axis. Defaults to 0.
        :param y: The y-coordinate of the center of the axis. Defaults to 0.
        :param length: The length of the axis. Defaults to 0.2.

:param Draws: 
        """
        super().annotate(image, **kwargs)
        draw_axis(image, self.rotation, vector.obj(x=x, y=y), length)
