import unittest

import numpy as np
import vector

from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.util.LinalgUtils import (
    project_pixel_to_point,
    project_pixels_to_points,
    project_point_to_pixel,
    project_points_to_pixels,
)
from visiongraph.util.MathUtils import decompose_transformation_matrix


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        intrinsic_matrix=np.array([[2.0, 0.0, 10.0], [0.0, 4.0, 20.0], [0.0, 0.0, 1.0]], dtype=float),
        distortion_coefficients=np.zeros(5, dtype=float),
    )


class LinalgUtilsTests(unittest.TestCase):
    def test_project_point_to_pixel_and_back_round_trip(self):
        intrinsics = _intrinsics()
        point = vector.obj(x=2.0, y=4.0, z=2.0)

        pixel = project_point_to_pixel(point, intrinsics)
        restored = project_pixel_to_point(pixel, depth=2.0, intrinsics=intrinsics)

        self.assertAlmostEqual(12.0, pixel.x)
        self.assertAlmostEqual(28.0, pixel.y)
        self.assertAlmostEqual(point.x, restored.x)
        self.assertAlmostEqual(point.y, restored.y)
        self.assertAlmostEqual(point.z, restored.z)

    def test_project_points_to_pixels_projects_each_row_independently(self):
        intrinsics = _intrinsics()
        points = np.array([[2.0, 4.0, 2.0], [6.0, 8.0, 2.0]], dtype=float)

        pixels = project_points_to_pixels(points, intrinsics)

        np.testing.assert_allclose(pixels, np.array([[12.0, 28.0], [16.0, 36.0]], dtype=float))

    def test_project_pixels_to_points_restores_vectorized_points(self):
        intrinsics = _intrinsics()
        pixels = np.array([[12.0, 28.0], [16.0, 36.0]], dtype=float)
        depth = np.array([2.0, 2.0], dtype=float)

        points = project_pixels_to_points(pixels, depth, intrinsics)

        np.testing.assert_allclose(points, np.array([[2.0, 4.0, 2.0], [6.0, 8.0, 2.0]], dtype=float))

    def test_decompose_transformation_matrix_separates_translation_and_scale(self):
        matrix = np.array(
            [
                [2.0, 0.0, 0.0, 5.0],
                [0.0, 3.0, 0.0, 6.0],
                [0.0, 0.0, 4.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        rotation, translation, scale = decompose_transformation_matrix(matrix)

        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(translation, np.array([5.0, 6.0, 7.0], dtype=float))
        np.testing.assert_allclose(scale, np.array([2.0, 3.0, 4.0], dtype=float))


if __name__ == "__main__":
    unittest.main()
