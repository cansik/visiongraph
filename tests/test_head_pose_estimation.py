import unittest

import cv2
from visiongraph import vg


class HeadPoseEstimationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.HeadPoseEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_adas_head_pose_estimator(self):
        self._test_model(vg.AdasHeadPoseEstimator())


if __name__ == '__main__':
    unittest.main()
