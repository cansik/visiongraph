import numpy as np
import vector

from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.DrawingUtils import draw_axis


class HeadPoseResult(BaseResult):
    def __init__(self, rotation: vector.Vector3D):
        self.rotation = rotation

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        super().annotate(image, **kwargs)
        draw_axis(image, self.rotation, vector.obj(x=x, y=y), length)
