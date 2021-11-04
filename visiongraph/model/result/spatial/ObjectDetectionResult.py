from typing import Optional

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.result.ClassificationResult import ClassificationResult
from visiongraph.model.tracker.Trackable import Trackable
from visiongraph.util.DrawingUtils import COLOR_SEQUENCE


class ObjectDetectionResult(ClassificationResult, Trackable):
    def __init__(self, class_id: int, class_name: str, score: float, bounding_box: BoundingBox2D):
        super().__init__(class_id, class_name, score)
        self._bounding_box = bounding_box

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None, **kwargs):
        super().annotate(image, **kwargs)

        h, w = image.shape[:2]
        color = COLOR_SEQUENCE[self.tracking_id % len(COLOR_SEQUENCE)]

        bbox = self.bounding_box
        cv2.rectangle(image, (round(bbox.x_min * w), round(bbox.y_min * h)),
                      (round((bbox.x_min + bbox.width) * w), round((bbox.y_min + bbox.height) * h)),
                      color, thickness=2)

        if not show_info:
            return

        if info_text is None:
            info_text = f"#{self.tracking_id}"

        cv2.putText(image, info_text,
                    (round(self.bounding_box.x_min * w) - 5,
                     round(self.bounding_box.y_min * h) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @property
    def bounding_box(self) -> BoundingBox2D:
        return self._bounding_box
