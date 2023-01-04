import unittest

import cv2
import visiongraph as vg


class FaceRecognitionEstimationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.FaceRecognitionEstimator):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_face_recognition_estimator(self):
        self._test_model(vg.FaceRecognitionEstimator())

    def test_face_reidentification_estimator_int8(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP16_INT8))

    def test_face_reidentification_estimator_fp16(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP16))

    def test_face_reidentification_estimator_fp32(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP32))


if __name__ == '__main__':
    unittest.main()
