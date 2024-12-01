import numpy as np

from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.BaseResult import BaseResult

INTRINSIC_MATRIX_NAME = "intrinsic_matrix"
DISTORTION_COEFFICIENTS_NAME = "distortion_coefficients"


class CameraPoseResult(BaseResult):
    """
    Represents the result of a camera pose estimation.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initializes the CameraPoseResult object with the given camera intrinsics.

        :param intrinsics: The intrinsics of the camera used for pose estimation.
        """
        self.intrinsics = intrinsics

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        """
        Adds annotations to the given image with the estimated camera pose.

        :param image: The input image.
        :param x: The x-coordinate of the annotation point. Defaults to 0.
        :param y: The y-coordinate of the annotation point. Defaults to 0.
        :param length: The size of the annotation box. Defaults to 0.2.

        """
        super().annotate(image, **kwargs)
