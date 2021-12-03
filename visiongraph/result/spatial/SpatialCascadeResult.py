from typing import Optional

import numpy as np

from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class SpatialCascadeResult(ObjectDetectionResult):
    def __init__(self, root_result: ObjectDetectionResult, **results: BaseResult):
        super().__init__(root_result.class_id, root_result.class_name, root_result.score, root_result.bounding_box)
        self.root_result = root_result
        self.results = results

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None, **kwargs):
        self.root_result.annotate(image, show_info, info_text, **kwargs)
        center = self.root_result.bounding_box.center

        for name, result in self.results.items():
            result.annotate(image, x=center.x, y=center.y, length=0.04, **kwargs)
