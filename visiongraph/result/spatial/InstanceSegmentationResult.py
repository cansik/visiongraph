from typing import Optional

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class InstanceSegmentationResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float,
                 mask: np.ndarray, bounding_box: BoundingBox2D):
        super().__init__(class_id, class_name, score, bounding_box)
        self.mask = mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, min_score: float = 0, **kwargs):
        if show_bounding_box:
            super().annotate(image, show_info, info_text, **kwargs)

        h, w = image.shape[:2]
        color = self.annotation_color

        # todo: real annotation of masks
        image[self.mask == 1] = [255, 0, 255]
