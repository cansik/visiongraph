import json

import cv2
import numpy as np

INTRINSIC_MATRIX_NAME = "intrinsic_matrix"
DISTORTION_COEFFICIENTS_NAME = "distortion_coefficients"


class CameraIntrinsics:
    def __init__(self, intrinsic_matrix: np.ndarray, distortion_coefficients: np.ndarray):
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coefficients = distortion_coefficients

    def save(self, path: str):
        data = {
            INTRINSIC_MATRIX_NAME: self.intrinsic_matrix.tolist(),
            DISTORTION_COEFFICIENTS_NAME: self.distortion_coefficients.tolist()
        }

        with open(path, "w") as file:
            json.dump(data, file, indent=4, sort_keys=True)

    @staticmethod
    def load(path: str) -> "CameraIntrinsics":
        with open(path, "r") as file:
            data = json.load(file)

        intrinsic_mat = np.array(data[INTRINSIC_MATRIX_NAME], dtype=float)
        distortion_coeff = np.array(data[DISTORTION_COEFFICIENTS_NAME], dtype=float)

        return CameraIntrinsics(intrinsic_mat, distortion_coeff)

    @staticmethod
    def load_from_file_storage(path: str):
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        intrinsic_mat = storage.getNode('Camera_Matrix').mat()
        distortion_coeff = storage.getNode('Distortion_Coefficients').mat()
        storage.release()

        return CameraIntrinsics(intrinsic_mat, distortion_coeff)

    @property
    def px(self) -> float:
        return self.intrinsic_matrix[0, 2]

    @property
    def py(self) -> float:
        return self.intrinsic_matrix[1, 2]

    @property
    def fx(self) -> float:
        return self.intrinsic_matrix[0, 0]

    @property
    def fy(self) -> float:
        return self.intrinsic_matrix[1, 1]

    def __repr__(self):
        return f"{CameraIntrinsics.__name__} (fx: {self.fx:.3f}, fy: {self.fy:.3f} px: {self.px:.3f}, py: {self.py:.3f})"
