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

        Args:
            class_id (int): Identifier for the detected class.
            class_name (str): Name of the detected class.
            score (float): Confidence score for the detection.
            mask (np.ndarray): Binary mask representing the segmented instance.
            bounding_box (BoundingBox2D): Bounding box surrounding the detected instance.
        """
        super().__init__(class_id, class_name, score, bounding_box)
        self.mask = mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 show_bounding_box: bool = True, use_class_color: bool = True, min_score: float = 0, **kwargs):
        """
        Annotates the given image with the instance segmentation result.

        Args:
            image (np.ndarray): The image to be annotated.
            show_info (bool, optional): Flag to display additional information. Defaults to True.
            info_text (Optional[str], optional): Custom text to display on the image. Defaults to None.
            show_bounding_box (bool, optional): Flag to display the bounding box. Defaults to True.
            use_class_color (bool, optional): Flag to use class color for the mask. Defaults to True.
            min_score (float, optional): Minimum score threshold for displaying annotations. Defaults to 0.
            **kwargs: Additional keyword arguments for further customization.
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

        Args:
            image (np.ndarray): The image to which the mask will be applied.

        Returns:
            np.ndarray: The resulting image with the mask applied.
        """
        return cv2.bitwise_and(image, image, mask=self.mask)
