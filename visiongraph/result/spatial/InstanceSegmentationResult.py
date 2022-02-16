from typing import Optional

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.DrawingUtils import COCO80_COLORS


class InstanceSegmentationResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float,
                 mask: np.ndarray, bounding_box: BoundingBox2D):
        super().__init__(class_id, class_name, score, bounding_box)
        self.mask = mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, use_class_color: bool = True, min_score: float = 0, **kwargs):
        if show_bounding_box:
            super().annotate(image, show_info, info_text, **kwargs)

        h, w = image.shape[:2]
        color = self.annotation_color

        if use_class_color:
            color = COCO80_COLORS[self.class_id]

        colored = np.zeros(image.shape, image.dtype)
        colored[:, :] = color
        colored_mask = cv2.bitwise_and(colored, colored, mask=self.mask)
        cv2.addWeighted(colored_mask, 0.75, image, 1.0, 0, image)

    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(image, image, mask=self.mask)
