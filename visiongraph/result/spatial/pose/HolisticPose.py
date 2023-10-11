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
                 face_score: float,
                 face_landmarks: vector.VectorNumpy4D,
                 right_hand_score: float,
                 right_hand_landmarks: vector.VectorNumpy4D,
                 left_hand_score: float,
                 left_hand_landmarks: vector.VectorNumpy4D,
                 segmentation_mask: Optional[np.ndarray] = None):
        super().__init__(pose_score, pose_landmarks)

        self.face = BlazeFaceMesh(face_score, face_landmarks)
        self.right_hand = BlazeHand(right_hand_score, right_hand_landmarks, Handedness.RIGHT)
        self.left_hand = BlazeHand(left_hand_score, left_hand_landmarks, Handedness.LEFT)

        self.segmentation_mask: Optional[np.ndarray] = segmentation_mask

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, use_class_color: bool = True, **kwargs):
        BlazePose.annotate(self, image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
        self.face.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
        self.right_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
        self.left_hand.annotate(image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
