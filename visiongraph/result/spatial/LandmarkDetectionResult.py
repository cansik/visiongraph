import copy
from typing import Optional, List, Tuple, Sequence, Union

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
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
                 connections: Optional[List[Tuple[int, int]]] = None,
                 marker_size: int = 3, stroke_width: int = 2, **kwargs):

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
                    cv2.line(image, point01, point02, color, stroke_width)

        # mark landmark joints
        for lm in self.landmarks:
            if lm.t < min_score:
                continue
            cv2.circle(image, (round(lm.x * w), round(lm.y * h)), marker_size, (0, 0, 255), -1)

    def map_coordinates(self, src_size: Union[Sequence[float], Size2D], dest_size: Union[Sequence[float], Size2D],
                        src_roi: Optional[BoundingBox2D] = None, dest_roi: Optional[BoundingBox2D] = None):
        src_width, src_height = src_size
        dest_width, dest_height = dest_size

        if src_roi is None:
            src_roi = BoundingBox2D(0, 0, src_width, src_height)

        if dest_roi is None:
            dest_roi = BoundingBox2D(0, 0, dest_width, dest_height)

        super().map_coordinates(src_size, dest_size, src_roi, dest_roi)

        for i, lm in enumerate(self.landmarks):
            x = lm.x * src_width
            x = (x - src_roi.x_min) / src_roi.width
            x = x * dest_roi.width + dest_roi.x_min
            x = x / dest_width

            y = lm.y * src_height
            y = (y - src_roi.y_min) / src_roi.height
            y = y * dest_roi.height + dest_roi.y_min
            y = y / dest_height

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
