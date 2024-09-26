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

        Args:
            intrinsics (CameraIntrinsics): The intrinsics of the camera used for pose estimation.
        """
        self.intrinsics = intrinsics

    def annotate(self, image: np.ndarray, x: float = 0, y: float = 0, length: float = 0.2, **kwargs):
        """
        Adds annotations to the given image with the estimated camera pose.

        Args:
            image (np.ndarray): The input image.
            x (float, optional): The x-coordinate of the annotation point. Defaults to 0.
            y (float, optional): The y-coordinate of the annotation point. Defaults to 0.
            length (float, optional): The size of the annotation box. Defaults to 0.2.

        Returns:
            None
        """
        super().annotate(image, **kwargs)
