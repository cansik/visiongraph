import unittest

import cv2

from visiongraph import vg


class FaceCascadeRegressionEstimationTests(unittest.TestCase):

    def setUp(self) -> None:
        self.network = vg.SpatialCascadeEstimator(vg.AdasFaceDetector.create())
        self.network.setup()

    def doCleanups(self) -> None:
        self.network.release()

    def _test_model(self, model: vg.FaceEmotionEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        result = self.network.process(image)[0]

        model.setup()
        model.process_detection(image, result)
        model.release()

    def test_face_affect_net_emotion_classifier_int8(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.INT8))

    def test_face_affect_net_emotion_classifier_fp16(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.FP16))

    def test_face_affect_net_emotion_classifier_fp32(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.FP32))

    def test_fer_plus_emotion_classifier(self):
        self._test_model(vg.FERPlusEmotionClassifier())


if __name__ == '__main__':
    unittest.main()
