from typing import Optional, List

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class LandmarkDetectionResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float, landmarks: vector.VectorNumpy4D):
        super().__init__(class_id, class_name, score, self._create_bounding_box(landmarks))
        self.landmarks: vector.VectorNumpy4D = landmarks

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, min_score: float = 0,
                 connections: Optional[List[int, int]] = None, **kwargs):

        if show_bounding_box:
            super().annotate(image, show_info, info_text, **kwargs)

        h, w = image.shape[:2]
        color = self.annotation_color

        # mark landmark joints
        for lm in self.landmarks:
            if lm.t < min_score:
                continue
            cv2.circle(image, (round(lm.x * w), round(lm.y * h)), 2, (0, 0, 255), -1)

        # draw connections
        if connections is None:
            return

        for ia, ib in connections:
            a: vector.Vector4D = self.landmarks[ia]
            b: vector.Vector4D = self.landmarks[ib]

            if a.t > min_score and b.t > min_score:
                point01 = (round(a.x * w), round(a.y * h))
                point02 = (round(b.x * w), round(b.y * h))
                cv2.line(image, point01, point02, color, 2)

    @staticmethod
    def _create_bounding_box(landmarks: vector.VectorNumpy4D) -> BoundingBox2D:
        xs = landmarks["x"]
        ys = landmarks["y"]

        x_min = np.min(xs)
        y_min = np.min(ys)
        x_max = np.max(xs)
        y_max = np.max(ys)

        return BoundingBox2D(x_min, y_min, x_max - x_min, y_max - y_min)
