from typing import Optional, Sequence

import numpy as np
import vector

from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.result.spatial.pose.BlazePose import BlazePose
from visiongraph.result.spatial.pose.PoseLandmarkResult import POSE_DETECTION_NAME, POSE_DETECTION_ID


class BlazePoseSegmentation(BlazePose, InstanceSegmentationResult):
    def __init__(self, score: float, landmarks: vector.VectorNumpy4D, mask: np.ndarray):
        BlazePose.__init__(self, score, landmarks)
        InstanceSegmentationResult.__init__(self, POSE_DETECTION_ID, POSE_DETECTION_NAME,
                                            score, mask, self.bounding_box)

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None,
                 show_bounding_box: bool = False, min_score: float = 0, use_class_color: bool = True, **kwargs):
        InstanceSegmentationResult.annotate(self, image, show_info, info_text, show_bounding_box,
                                            use_class_color, min_score, **kwargs)
        BlazePose.annotate(self, image, show_info, info_text, color, show_bounding_box, min_score, **kwargs)
