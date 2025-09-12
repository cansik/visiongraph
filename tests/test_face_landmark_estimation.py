import unittest

import cv2

from tests.utils_for_testing import run_estimator_test
from visiongraph import vg


class FaceLandmarkEstimationTests(unittest.TestCase):
    def _test_model(self, model: vg.FaceLandmarkEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        run_estimator_test(model, image, self._testMethodName)

    def test_mediapipe_face_landmark_detector(self):
        self._test_model(vg.MediaPipeFaceDetector())

    def test_mediapipe_face_mesh_landmark_detector(self):
        self._test_model(vg.MediaPipeFaceMeshEstimator())

    def test_regression_landmark_estimator(self):
        self._test_model(vg.RegressionLandmarkEstimator())


if __name__ == "__main__":
    unittest.main()
