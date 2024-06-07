from typing import Optional, Sequence

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class OrientedObjectDetectionResult(ObjectDetectionResult):
    def __init__(self, class_id: int, class_name: str, score: float, bounding_box: BoundingBox2D, theta: float):
        super().__init__(class_id, class_name, score, bounding_box)
        self.theta = theta

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, **kwargs):
        h, w = image.shape[:2]
        color = self.annotation_color if color is None else color

        self.draw_oriented_bbox2d(image, self.bounding_box, color)

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

    def draw_oriented_bbox2d(self, image: np.ndarray, bbox: BoundingBox2D, color: Sequence[int], thickness: int = 2):
        h, w = image.shape[:2]

        center = bbox.center
        self.draw_oriented_bbox(image, round(center.x * w), round(center.y * h),
                                round(bbox.width * w), round(bbox.height * h),
                                self.theta, color, thickness)

    def draw_oriented_bbox(self, image: np.ndarray, cx: int, cy: int, w: int, h: int, theta: float,
                           color: Sequence[int], thickness: int = 2) -> None:
        # Calculate the corner points of the bounding box
        rect = ((cx, cy), (w, h), theta)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw the bounding box on the image
        cv2.drawContours(image, [box], 0, color, thickness)

        # Mark start of orientation
        cv2.circle(image, box[0], max(1, round(w * 0.1)), color=(255, 0, 255), thickness=-1)
        cv2.line(image, box[0], box[1], color=(255, 0, 255), thickness=2)
