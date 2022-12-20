import copy
import numbers
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
                 marker_size: int = 3, marker_type: Optional[int] = None, stroke_width: int = 2,
                 landmark_colors: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None, **kwargs):

        if show_bounding_box:
            super().annotate(image, show_info, info_text, color, **kwargs)

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
        if landmark_colors is None:
            landmark_colors = [(0, 0, 255)]

        if isinstance(landmark_colors, Sequence):
            if len(landmark_colors) == 0:
                raise Exception("Landmark colors can not be empty!")

            if isinstance(landmark_colors[0], numbers.Real):
                landmark_colors = [landmark_colors]

        for i, lm in enumerate(self.landmarks):
            if lm.t < min_score:
                continue
            lm_color = landmark_colors[i % len(landmark_colors)]
            position = (round(lm.x * w), round(lm.y * h))

            if marker_type is None:
                cv2.circle(image, position, marker_size, lm_color, -1)
            else:
                cv2.drawMarker(image, position, lm_color, markerType=marker_type, markerSize=marker_size)

    def map_coordinates(self, src_size: Union[Sequence[float], Size2D], dest_size: Union[Sequence[float], Size2D],
                        src_roi: Optional[BoundingBox2D] = None, dest_roi: Optional[BoundingBox2D] = None):
        src_width, src_height = src_size
        dest_width, dest_height = dest_size

        if src_roi is None:
            src_roi = BoundingBox2D(0, 0, src_width, src_height)

        if dest_roi is None:
            dest_roi = BoundingBox2D(0, 0, dest_width, dest_height)

        super().map_coordinates(src_size, dest_size, src_roi, dest_roi)

        self.landmarks.x[:] = ((((self.landmarks.x * src_width) - src_roi.x_min) / src_roi.width)
                               * dest_roi.width + dest_roi.x_min) / dest_width
        self.landmarks.y[:] = ((((self.landmarks.y * src_height) - src_roi.y_min) / src_roi.height)
                               * dest_roi.height + dest_roi.y_min) / dest_height

    @staticmethod
    def _create_bounding_box(landmarks: vector.VectorNumpy4D) -> BoundingBox2D:
        xs = np.ma.masked_equal(landmarks["x"], 0.0, copy=False)
        ys = np.ma.masked_equal(landmarks["y"], 0.0, copy=False)

        x_min = xs.min()
        y_min = ys.min()
        x_max = xs.max()
        y_max = ys.max()

        return BoundingBox2D(x_min, y_min, x_max - x_min, y_max - y_min)
