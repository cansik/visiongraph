from typing import Optional, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.pose.BlazePose import BlazePose


class HolisticPose(BlazePose):
    """
    A class to encapsulate holistic pose information, including pose score,
    landmarks, facial mesh, and hand landmarks.
    """

    def __init__(self, pose_score: float,
                 pose_landmarks: vector.VectorNumpy4D,
                 segmentation_mask: Optional[np.ndarray] = None):
        """
        Initializes a HolisticPose object with specified score, landmarks, and optional segmentation mask.

        Args:
            pose_score (float): The score representing the confidence of the detected pose.
            pose_landmarks (vector.VectorNumpy4D): The landmarks of the pose as a 4D vector.
            segmentation_mask (Optional[np.ndarray], optional): The optional mask for segmenting the pose. Defaults to None.
        """
        super().__init__(pose_score, pose_landmarks)

        self.face: Optional[BlazeFaceMesh] = None
        self.right_hand: Optional[BlazeHand] = None
        self.left_hand: Optional[BlazeHand] = None

        self.segmentation_mask: Optional[np.ndarray] = segmentation_mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, use_class_color: bool = True,
                 pose_only: bool = False, **kwargs):
        """
        Annotates the given image with the pose, hand, and face information.

        Args:
            image (np.ndarray): The image to be annotated.
            show_info (bool, optional): Flag to control viewing of additional information. Defaults to True.
            info_text (Optional[str], optional): Additional information text to display. Defaults to None.
            color (Optional[Sequence[int]], optional): Color for annotation. Defaults to None.
            show_bounding_box (bool, optional): Flag to show bounding boxes. Defaults to False.
            min_score (float, optional): The minimum score threshold for annotations. Defaults to 0.
            use_class_color (bool, optional): Flag to use class-specific colors. Defaults to True.
            pose_only (bool, optional): Flag to indicate if only pose should be annotated. Defaults to False.
            **kwargs: Additional keyword arguments for further customization of annotation.
        """
        BlazePose.annotate(self, image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if pose_only:
            return

        if self.face is not None:
            self.face.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if self.right_hand is not None:
            self.right_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if self.left_hand is not None:
            self.left_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
