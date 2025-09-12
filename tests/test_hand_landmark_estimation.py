import unittest

import cv2

from tests.utils_for_testing import run_estimator_test
from visiongraph import vg


class HandLandmarkTests(unittest.TestCase):
    def _test_model(self, model: vg.HandLandmarkEstimator):
        image = cv2.imread("assets/hands-pexels-ketut-subiyanto-4126739.jpg")
        run_estimator_test(model, image, self._testMethodName)

    def test_mediapipe_hand_estimator(self):
        self._test_model(vg.MediaPipeHandEstimator())

    def test_openpose_hand_estimator(self):
        self._test_model(vg.OpenPoseHandEstimator())


if __name__ == "__main__":
    unittest.main()
