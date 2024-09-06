import unittest

import cv2
from visiongraph import vg


class FaceLandmarkEstimationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.FaceLandmarkEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_mediapipe_face_landmark_detector(self):
        self._test_model(vg.MediaPipeFaceDetector())

    def test_mediapipe_face_mesh_landmark_detector(self):
        self._test_model(vg.MediaPipeFaceMeshEstimator())

    def test_regression_landmark_estimator(self):
        self._test_model(vg.RegressionLandmarkEstimator())


if __name__ == '__main__':
    unittest.main()
