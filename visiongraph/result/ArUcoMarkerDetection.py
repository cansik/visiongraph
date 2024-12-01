from typing import Sequence

import cv2
import numpy as np
import vector
from vector import Vector2D

from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.VectorUtils import vector_to_array


class ArUcoMarkerDetection(BaseResult):
    """
    Represents an object detected using the ARUCO marker detection algorithm.
    """

    def __init__(self, marker_id: int,
                 top_left: Vector2D, top_right: Vector2D,
                 bottom_right: Vector2D, bottom_left: Vector2D):
        """
        Initializes the ArUcoMarkerDetection object with marker ID and bounding box coordinates.

        :param marker_id: The unique identifier of the detected marker.
        :param top_left: The top-left corner of the bounding box.
        :param top_right: The top-right corner of the bounding box.
        :param bottom_right: The bottom-right corner of the bounding box.
        :param bottom_left: The bottom-left corner of the bounding box.
        """
        self.marker_id = marker_id

        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left

    def annotate(self, image: np.ndarray,
                 color: Sequence[int] = (0, 255, 0),
                 thickness: int = 1,
                 **kwargs):
        """
        Draws the bounding box and marker on the given image.

        :param image: The input image.
        :param color: The color to use for drawing. Defaults to green.
        :param thickness: The line thickness. Defaults to 1.
        """
        super().annotate(image, **kwargs)

        vertices = np.array([vector_to_array(self.top_left),
                             vector_to_array(self.top_right),
                             vector_to_array(self.bottom_right),
                             vector_to_array(self.bottom_left)], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=thickness)

        center = self.center
        cv2.drawMarker(image, (round(center.x), round(center.y)), color, markerType=cv2.MARKER_CROSS)

    @property
    def center(self) -> vector.Vector2D:
        """
        Calculates the center of the bounding box.

        :return: The center coordinates.
        """
        return vector.obj(x=(self.top_left.x + self.bottom_right.x) / 2.0,
                          y=(self.top_left.y + self.bottom_right.y) / 2.0)
