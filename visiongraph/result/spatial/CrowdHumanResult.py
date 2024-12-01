from typing import Optional, Sequence

import numpy as np

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class CrowdHumanResult(ObjectDetectionResult):
    """
    Represents the result of detecting a crowd human, including information
    about the person and their head if available.
    """

    def __init__(self, person: ObjectDetectionResult, head: Optional[ObjectDetectionResult]):
        """
        Initializes the CrowdHumanResult with the detected person and an optional head detection.

        :param person: The result of the object detection for the person.
        :param head: The result of the object detection for the head, if detected.
        """
        super().__init__(person.class_id, person.class_name, person.score, person.bounding_box)
        self.head = head

    def annotate(self, image: np.ndarray, show_info: bool = True, info_text: Optional[str] = None,
                 color: Optional[Sequence[int]] = None, **kwargs):
        """
        Annotates the given image with the detected person's and their head's information.

        :param image: The image to annotate.
        :param show_info: Whether to display additional info on the image.
        :param info_text: Custom text to display.
        :param color: Color for the annotations, specified as a sequence of RGB values.
        :param **kwargs: Additional keyword arguments for customization.
        """
        super().annotate(image, show_info, info_text, color, **kwargs)

        if self.head is None:
            return

        self.head.tracking_id = self.tracking_id
        self.head.annotate(image)
