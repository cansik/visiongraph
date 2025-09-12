import unittest

import cv2

from tests.utils_for_testing import run_estimator_test
from visiongraph import vg


class HeadPoseEstimationTests(unittest.TestCase):
    def _test_model(self, model: vg.HeadPoseEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        run_estimator_test(model, image, self._testMethodName)

    def test_adas_head_pose_estimator(self):
        self._test_model(vg.AdasHeadPoseEstimator())


if __name__ == "__main__":
    unittest.main()
