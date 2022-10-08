from typing import Union, Dict

import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

PADDING_BOX_OUTPUT_NAME = "padding-box"
IMAGE_BOX_OUTPUT_NAME = "image-box"


class VisionEngineOutput(Dict[str, Union[np.ndarray, BoundingBox2D]]):
    @property
    def padding_box(self) -> BoundingBox2D:
        return self[PADDING_BOX_OUTPUT_NAME]

    @padding_box.setter
    def padding_box(self, box: BoundingBox2D):
        self[PADDING_BOX_OUTPUT_NAME] = box

    @property
    def image_box(self) -> BoundingBox2D:
        return self[IMAGE_BOX_OUTPUT_NAME]

    @image_box.setter
    def image_box(self, box: BoundingBox2D):
        self[IMAGE_BOX_OUTPUT_NAME] = box
