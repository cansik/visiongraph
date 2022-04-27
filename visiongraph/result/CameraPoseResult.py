import json

import numpy as np

from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.BaseResult import BaseResult

INTRINSIC_MATRIX_NAME = "intrinsic_matrix"
DISTORTION_COEFFICIENTS_NAME = "distortion_coefficients"


class CameraPoseResult(BaseResult):
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        super().annotate(image, **kwargs)
