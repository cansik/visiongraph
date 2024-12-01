from typing import Union, Dict

import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D

PADDING_BOX_OUTPUT_NAME = "padding-box"
IMAGE_SIZE_OUTPUT_NAME = "image-box"


class VisionEngineOutput(Dict[str, Union[np.ndarray, BoundingBox2D, Size2D]]):
    """
    Represents the output of a vision engine, containing bounding boxes and image sizes.
    """

    @property
    def padding_box(self) -> BoundingBox2D:
        """
        Gets the 2D bounding box representing the padded output.

        :return: The padded 2D bounding box.
        """
        return self[PADDING_BOX_OUTPUT_NAME]

    @padding_box.setter
    def padding_box(self, box: BoundingBox2D):
        """
        Sets the 2D bounding box representing the padded output.

        :param box: The new padded 2D bounding box.
        """
        self[PADDING_BOX_OUTPUT_NAME] = box

    @property
    def image_size(self) -> Size2D:
        """
        Gets the 2D size representing the original image dimensions.

        :return: The original 2D size of the image.
        """
        return self[IMAGE_SIZE_OUTPUT_NAME]

    @image_size.setter
    def image_size(self, size: Size2D):
        """
        Sets the 2D size representing the original image dimensions.

        :param size: The new original 2D size of the image.
        """
        self[IMAGE_SIZE_OUTPUT_NAME] = size
