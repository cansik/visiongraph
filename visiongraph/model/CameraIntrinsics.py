import json

import cv2
import numpy as np

INTRINSIC_MATRIX_NAME = "intrinsic_matrix"
DISTORTION_COEFFICIENTS_NAME = "distortion_coefficients"


class CameraIntrinsics:
    """
    Represents the intrinsic parameters of a camera.
    """

    def __init__(self, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray):
        """
        Initializes the CameraIntrinsics object with the given intrinsic and distortion parameters.

        :param intrinsic_matrix: The 3x4 intrinsic matrix.
        :param distortion_coefficients: The distortion coefficients.
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    def save(self, path: str):
        """
        Saves the camera intrinsics to a JSON file.

        :param path: The file path to save the data to.
        """
        data = {
            INTRINSIC_MATRIX_NAME: self.intrinsic_matrix.tolist(),
            DISTORTION_COEFFICIENTS_NAME: self.distortion_coefficients.tolist()
        }

        with open(path, "w") as file:
            json.dump(data, file, indent=4, sort_keys=True)

    @staticmethod
    def load(path: str) -> "CameraIntrinsics":
        """
        Loads the camera intrinsics from a JSON file.

        :param path: The file path to load the data from.

        :return: The loaded CameraIntrinsics object.
        """
        with open(path, "r") as file:
            data = json.load(file)

        intrinsic_mat = np.array(data[INTRINSIC_MATRIX_NAME], dtype=float)
        distortion_coeff = np.array(data[DISTORTION_COEFFICIENTS_NAME], dtype=float)

        return CameraIntrinsics(intrinsic_mat, distortion_coeff)

    @staticmethod
    def load_from_file_storage(path: str):
        """
        Loads the camera intrinsics from a file stored in OpenCV's storage format.

        :param path: The file path to load the data from.

        :return: The loaded CameraIntrinsics object.
        """
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        intrinsic_mat = storage.getNode('Camera_Matrix').mat()
        distortion_coeff = storage.getNode('Distortion_Coefficients').mat()
        storage.release()

        return CameraIntrinsics(intrinsic_mat, distortion_coeff)

    @property
    def px(self) -> float:
        """
        Gets the pixel distance from the principal point.

        :return: The pixel distance.
        """
        return self.intrinsic_matrix[0, 2]

    @property
    def py(self) -> float:
        """
        Gets the pixel distance from the principal point (y-axis).

        :return: The pixel distance.
        """
        return self.intrinsic_matrix[1, 2]

    @property
    def fx(self) -> float:
        """
        Gets the focal length in the x-direction.

        :return: The focal length.
        """
        return self.intrinsic_matrix[0, 0]

    @property
    def fy(self) -> float:
        """
        Gets the focal length in the y-direction.

        :return: The focal length.
        """
        return self.intrinsic_matrix[1, 1]

    def __repr__(self):
        """
        Returns a string representation of the CameraIntrinsics object.

        :return: A string representation of the object.
        """
        return f"{CameraIntrinsics.__name__} (fx: {self.fx:.3f}, fy: {self.fy:.3f} px: {self.px:.3f}, py: {self.py:.3f})"
