from typing import Optional, Sequence

import numpy as np

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class CrowdHumanResult(ObjectDetectionResult):

    def __init__(self, person: ObjectDetectionResult, head: Optional[ObjectDetectionResult]):
        super().__init__(person.class_id, person.class_name, person.score, person.bounding_box)
        self.head = head

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, **kwargs):
        super().annotate(image, show_info, info_text, color, **kwargs)

        if self.head is None:
            return

        self.head.tracking_id = self.tracking_id
        self.head.annotate(image)
