from typing import Optional, Sequence, Union

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.model.tracker.Trackable import Trackable
from visiongraph.result.ClassificationResult import ClassificationResult
from visiongraph.util.DrawingUtils import COLOR_SEQUENCE, draw_bbox


class ObjectDetectionResult(ClassificationResult, Trackable):
    """
    Represents the result of an object detection task, including its classification and bounding box.
    """

    def __init__(self, class_id: int, class_name: str, score: float, bounding_box: BoundingBox2D):
        """
        Initializes an ObjectDetectionResult instance.

        :param class_id: The ID of the detected class.
        :param class_name: The name of the detected class.
        :param score: The confidence score of the detection.
        :param bounding_box: The bounding box enclosing the detected object.
        """
        super().__init__(class_id, class_name, score)

        self._tracking_id = -1
        self._staleness = 0
        self._bounding_box = bounding_box

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, **kwargs):
        """
        Annotates an image with the object detection result.

        :param image: The image to annotate.
        :param show_info: Flag indicating whether to display additional information.
        :param info_text: Custom text to display on the annotation.
        :param color: Color for the annotation box.
        :param **kwargs: Additional keyword arguments for annotation.
        """
        super().annotate(image, **kwargs)

        h, w = image.shape[:2]
        color = self.annotation_color if color is None else color

        draw_bbox(image, self.bounding_box, color=color)

        if not show_info:
            return

        if info_text is None:
            if self.tracking_id >= 0:
                info_text = f"#{self.tracking_id} "
            else:
                info_text = ""

            if self.class_name is not None:
                info_text += f"{self.class_name}"

        cv2.putText(image, info_text,
                    (round(self.bounding_box.x_min * w) - 5,
                     round(self.bounding_box.y_min * h) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @property
    def bounding_box(self) -> BoundingBox2D:
        """
        Gets the bounding box of the detected object.

        :return: The bounding box enclosing the detected object.
        """
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, box: BoundingBox2D):
        """
        Sets the bounding box of the detected object.

        :param box: The new bounding box to set.
        """
        self._bounding_box = box

    @property
    def annotation_color(self):
        """
        Gets the color used for annotation based on the tracking ID.

        :return: The color RGB values for annotation.
        """
        return COLOR_SEQUENCE[self.tracking_id % len(COLOR_SEQUENCE)]

    @property
    def tracking_id(self) -> int:
        """
        Gets the tracking ID of the object.

        :return: The tracking ID of the object.
        """
        return self._tracking_id

    @tracking_id.setter
    def tracking_id(self, value: int):
        """
        Sets the tracking ID of the object.

        :param value: The new tracking ID to set.
        """
        self._tracking_id = value

    @property
    def staleness(self) -> int:
        """
        Gets the staleness of the object.

        :return: The staleness value indicating how outdated the detection is.
        """
        return self._staleness

    @staleness.setter
    def staleness(self, value: int):
        """
        Sets the staleness of the object.

        :param value: The new staleness value to set.
        """
        self._staleness = value

    @property
    def is_stale(self) -> bool:
        """
        Checks whether the object detection result is considered stale.

        :return: True if the staleness is greater than 0, otherwise False.
        """
        return self._staleness > 0

    def map_coordinates(self, src_size: Union[Sequence[float], Size2D], dest_size: Union[Sequence[float], Size2D],
                        src_roi: Optional[BoundingBox2D] = None, dest_roi: Optional[BoundingBox2D] = None):
        """
        Maps the coordinates of the bounding box from source size to destination size.

        :param src_size: The size of the source.
        :param dest_size: The size of the destination.
        :param src_roi: The region of interest in the source.
        :param dest_roi: The region of interest in the destination.
        """
        bbox = self._bounding_box

        src_width, src_height = src_size
        dest_width, dest_height = dest_size

        if src_roi is None:
            src_roi = BoundingBox2D(0, 0, src_width, src_height)

        if dest_roi is None:
            dest_roi = BoundingBox2D(0, 0, dest_width, dest_height)

        x = bbox.x_min * src_width
        x = (x - src_roi.x_min) / src_roi.width
        x = x * dest_roi.width + dest_roi.x_min
        x = x / dest_width

        y = bbox.y_min * src_height
        y = (y - src_roi.y_min) / src_roi.height
        y = y * dest_roi.height + dest_roi.y_min
        y = y / dest_height

        bbox.x_min = x
        bbox.y_min = y

        bbox.width = ((bbox.width * src_width) / src_roi.width) * dest_roi.width / dest_width
        bbox.height = (bbox.height * src_height / src_roi.height) * dest_roi.height / dest_height
