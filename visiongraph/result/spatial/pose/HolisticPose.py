from typing import Optional, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.result.spatial.pose.BlazePose import BlazePose


class HolisticPose(BlazePose):
    def __init__(self, pose_score: float,
                 pose_landmarks: vector.VectorNumpy4D,
                 segmentation_mask: Optional[np.ndarray] = None):
        super().__init__(pose_score, pose_landmarks)

        self.face: Optional[BlazeFaceMesh] = None
        self.right_hand: Optional[BlazeHand] = None
        self.left_hand: Optional[BlazeHand] = None

        self.segmentation_mask: Optional[np.ndarray] = segmentation_mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, use_class_color: bool = True,
                 pose_only: bool = False, **kwargs):
        BlazePose.annotate(self, image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if pose_only:
            return

        if self.face is not None:
            self.face.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if self.right_hand is not None:
            self.right_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)

        if self.left_hand is not None:
            self.left_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
