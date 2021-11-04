from typing import Optional

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class LandmarkDetectionResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(class_id, class_name, score, self._create_bounding_box(landmarks))
        self.landmarks = landmarks

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, **kwargs):

        if show_bounding_box:
            super().annotate(image, show_info, info_text, **kwargs)

        # mark landmarks
        h, w = image.shape[:2]
        for lm in self.landmarks:
            cv2.circle(image, (round(lm.x * w), round(lm.y * h)), 2, (0, 0, 255), -1)

    @staticmethod
    def _create_bounding_box(landmarks: vector.VectorNumpy4D) -> BoundingBox2D:
        xs = landmarks["x"]
        ys = landmarks["y"]

        x_min = np.min(xs)
        y_min = np.min(ys)
        x_max = np.max(xs)
        y_max = np.max(ys)

        return BoundingBox2D(x_min, y_min, x_max - x_min, y_max - y_min)
