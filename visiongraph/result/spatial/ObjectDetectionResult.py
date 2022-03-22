from typing import Optional, Sequence

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ClassificationResult import ClassificationResult
from visiongraph.model.tracker.Trackable import Trackable
from visiongraph.util.DrawingUtils import COLOR_SEQUENCE, draw_bbox


class ObjectDetectionResult(ClassificationResult, Trackable):
    def __init__(self, class_id: int, class_name: str, score: float, bounding_box: BoundingBox2D):
        super().__init__(class_id, class_name, score)

        self._tracking_id = -1
        self._bounding_box = bounding_box

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, **kwargs):
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
        return self._bounding_box

    @property
    def annotation_color(self):
        return COLOR_SEQUENCE[self.tracking_id % len(COLOR_SEQUENCE)]

    @property
    def tracking_id(self) -> int:
        return self._tracking_id

    @tracking_id.setter
    def tracking_id(self, value: int):
        self._tracking_id = value


