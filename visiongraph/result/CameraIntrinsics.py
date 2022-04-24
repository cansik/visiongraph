import numpy as np

from visiongraph.result.BaseResult import BaseResult


class CameraIntrinsics(BaseResult):
    def __init__(self, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray):
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    def annotate(self, image: np.ndarray, **kwargs):
        super().annotate(image, **kwargs)
