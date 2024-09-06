import unittest

import cv2
from visiongraph import vg
from visiongraph.util import OSUtils


class HandLandmarkTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.HandLandmarkEstimator):
        image = cv2.imread("assets/hands-pexels-ketut-subiyanto-4126739.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_mediapipe_hand_estimator(self):
        self._test_model(vg.MediaPipeHandEstimator())

    def test_openpose_hand_estimator(self):
        self._test_model(vg.OpenPoseHandEstimator())


if __name__ == '__main__':
    unittest.main()
