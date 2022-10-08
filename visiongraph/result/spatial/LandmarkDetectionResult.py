import copy
from typing import Optional, List, Tuple, Sequence

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class LandmarkDetectionResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float,
                 landmarks: vector.VectorNumpy4D, bounding_box: Optional[BoundingBox2D] = None):
        if bounding_box is None:
            bounding_box = self._create_bounding_box(landmarks)

        ObjectDetectionResult.__init__(self, class_id, class_name, score, bounding_box)
        self.landmarks: vector.VectorNumpy4D = landmarks

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0,
                 connections: Optional[List[Tuple[int, int]]] = None, **kwargs):

        if show_bounding_box:
            super().annotate(image, show_info, info_text, **kwargs)

        h, w = image.shape[:2]
        color = self.annotation_color if color is None else color

        # draw connections
        if connections is not None:
            for ia, ib in connections:
                a: vector.Vector4D = self.landmarks[ia]  # type: ignore
                b: vector.Vector4D = self.landmarks[ib]  # type: ignore

                if a.t > min_score and b.t > min_score:
                    point01 = (round(a.x * w), round(a.y * h))
                    point02 = (round(b.x * w), round(b.y * h))
                    cv2.line(image, point01, point02, color, 2)

        # mark landmark joints
        for lm in self.landmarks:
            if lm.t < min_score:
                continue
            cv2.circle(image, (round(lm.x * w), round(lm.y * h)), 3, (0, 0, 255), -1)

    def map_coordinates(self, src_box: BoundingBox2D, dst_box: BoundingBox2D):
        super().map_coordinates(src_box, dst_box)

        for i, lm in enumerate(self.landmarks):
            x = ((lm.x * src_box.width) - dst_box.x_min) / dst_box.width
            y = ((lm.y * src_box.height) - dst_box.y_min) / dst_box.height
            self.landmarks.x[i] = x
            self.landmarks.y[i] = y

    @staticmethod
    def _create_bounding_box(landmarks: vector.VectorNumpy4D) -> BoundingBox2D:
        xs = np.ma.masked_equal(landmarks["x"], 0.0, copy=False)
        ys = np.ma.masked_equal(landmarks["y"], 0.0, copy=False)

        x_min = xs.min()
        y_min = ys.min()
        x_max = xs.max()
        y_max = ys.max()

        return BoundingBox2D(x_min, y_min, x_max - x_min, y_max - y_min)
