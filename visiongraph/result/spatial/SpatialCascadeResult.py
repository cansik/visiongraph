from typing import Optional

import numpy as np

from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class SpatialCascadeResult(ObjectDetectionResult):
    """
    Represents the result of a spatial cascade object detection process.

    Inherits from ObjectDetectionResult and holds additional results
    corresponding to the detection process.
    """

    def __init__(self, root_result: ObjectDetectionResult, **results: BaseResult):
        """
        Initializes the SpatialCascadeResult with a root object detection result
        and any additional results.

        :param root_result: The primary result object
        :param **results: Additional results associated with this cascade.
        """
        super().__init__(root_result.class_id, root_result.class_name, root_result.score, root_result.bounding_box)
        self.root_result = root_result
        self.results = results

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None, **kwargs):
        """
        Annotates the given image with the results of the object detection.

        :param image: The image to be annotated.
        :param show_info: Flag to determine if additional info should be displayed. Defaults to True.
        :param info_text: Additional text info to show on the image. Defaults to None.
        :param **kwargs: Additional keyword arguments for further customization.
        """
        self.root_result.annotate(image, show_info, info_text, **kwargs)
        center = self.root_result.bounding_box.center

        for name, result in self.results.items():
            result.annotate(image, x=center.x, y=center.y, length=0.04, **kwargs)
