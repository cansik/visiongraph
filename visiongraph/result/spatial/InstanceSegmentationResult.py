from typing import Optional

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.DrawingUtils import COCO80_COLORS


class InstanceSegmentationResult(ObjectDetectionResult):
    """
    Represents the result of instance segmentation, containing class information,
    a mask for the segmented area, and the bounding box for the instance.
    """

    def __init__(self, class_id: int, class_name: str, score: float,
                 mask: np.ndarray, bounding_box: BoundingBox2D):
        """
        Initializes an InstanceSegmentationResult with class details, segmentation mask,
        and corresponding bounding box.

        :param class_id: Identifier for the detected class.
        :param class_name: Name of the detected class.
        :param score: Confidence score for the detection.
        :param mask: Binary mask representing the segmented instance.
        :param bounding_box: Bounding box surrounding the detected instance.
        """
        super().__init__(class_id, class_name, score, bounding_box)
        self.mask = mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, use_class_color: bool = True, min_score: float = 0, **kwargs):
        """
        Annotates the given image with the instance segmentation result.

        :param image: The image to be annotated.
        :param show_info: Flag to display additional information. Defaults to True.
        :param info_text: Custom text to display on the image. Defaults to None.
        :param show_bounding_box: Flag to display the bounding box. Defaults to True.
        :param use_class_color: Flag to use class color for the mask. Defaults to True.
        :param min_score: Minimum score threshold for displaying annotations. Defaults to 0.
        :param **kwargs: Additional keyword arguments for further customization.
        """
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
        """
        Applies the segmentation mask to the input image, returning the masked region.

        :param image: The image to which the mask will be applied.

        :return: The resulting image with the mask applied.
        """
        return cv2.bitwise_and(image, image, mask=self.mask)
