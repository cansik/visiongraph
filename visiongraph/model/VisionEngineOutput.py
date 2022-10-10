from typing import Union, Dict

import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D

PADDING_BOX_OUTPUT_NAME = "padding-box"
IMAGE_SIZE_OUTPUT_NAME = "image-box"


class VisionEngineOutput(Dict[str, Union[np.ndarray, BoundingBox2D, Size2D]]):
    @property
    def padding_box(self) -> BoundingBox2D:
        return self[PADDING_BOX_OUTPUT_NAME]

    @padding_box.setter
    def padding_box(self, box: BoundingBox2D):
        self[PADDING_BOX_OUTPUT_NAME] = box

    @property
    def image_size(self) -> Size2D:
        return self[IMAGE_SIZE_OUTPUT_NAME]

    @image_size.setter
    def image_size(self, size: Size2D):
        self[IMAGE_SIZE_OUTPUT_NAME] = size
