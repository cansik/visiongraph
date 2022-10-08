from typing import Union, Dict

import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

PADDING_BOX_OUTPUT_NAME = "padding-box"


class VisionEngineOutput(Dict[str, Union[np.ndarray, BoundingBox2D]]):
    @property
    def padding_box(self) -> BoundingBox2D:
        return self[PADDING_BOX_OUTPUT_NAME]

    @padding_box.setter
    def padding_box(self, box: BoundingBox2D):
        self[PADDING_BOX_OUTPUT_NAME] = box
