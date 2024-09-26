from typing import Optional, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.PoseLandmarkResult import POSE_DETECTION_NAME, POSE_DETECTION_ID


class BlazePoseSegmentation(BlazePose, InstanceSegmentationResult):
    """
    A class that combines BlazePose functionality with instance segmentation results.

    Inherits from:
        BlazePose: For pose landmark detection.
        InstanceSegmentationResult: For handling segmentation masks and results.
    """

    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, mask: np.ndarray):
        """
        Initializes a BlazePoseSegmentation object with a score, landmarks, and segmentation mask.

        Args:
            score (float): The confidence score of the detected pose.
            landmarks (vector.VectorNumpy4D): The pose landmarks as a 4D vector.
            mask (np.ndarray): The segmentation mask associated with the instance.
        """
        BlazePose.__init__(self, score, landmarks)
        InstanceSegmentationResult.__init__(self, POSE_DETECTION_ID, POSE_DETECTION_NAME,
                                            score, mask, self.bounding_box)

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, use_class_color: bool = True, **kwargs):
        """
        Annotates the given image with pose landmarks and segmentation information.

        Args:
            image (np.ndarray): The image to be annotated.
            show_info (bool, optional): Whether to display additional information on the image. Defaults to True.
            info_text (Optional[str], optional): Additional text to display on the image. Defaults to None.
            color (Optional[Sequence[int]], optional): The color for the annotation. Defaults to None.
            show_bounding_box (bool, optional): Whether to draw a bounding box around the detected instance. Defaults to False.
            min_score (float, optional): Minimum score threshold for displaying annotations. Defaults to 0.
            use_class_color (bool, optional): Whether to use the class color for annotations. Defaults to True.
            **kwargs: Additional keyword arguments for further customization.
        """
        InstanceSegmentationResult.annotate(self, image, show_info, info_text, show_bounding_box,
                                            use_class_color, min_score, **kwargs)
        BlazePose.annotate(self, image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
