import unittest

import cv2

from tests.utils_for_testing import save_annotation_image
from visiongraph import vg


class FaceCascadeRegressionEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = vg.SpatialCascadeEstimator(vg.AdasFaceDetector.create())
        self.network.setup()

    def doCleanups(self) -> None:
        self.network.release()

    def _test_model(self, model: vg.FaceEmotionEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        face_result = self.network.process(image)[0]

        model.setup()
        result = model.process_detection(image, face_result)
        result.annotate(image)
        save_annotation_image(image, self._testMethodName)
        model.release()

    def test_face_affect_net_emotion_classifier_int8(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.INT8))

    def test_face_affect_net_emotion_classifier_fp16(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.FP16))

    def test_face_affect_net_emotion_classifier_fp32(self):
        self._test_model(vg.AffectNetEmotionClassifier(vg.ModelPrecision.FP32))

    def test_fer_plus_emotion_classifier(self):
        self._test_model(vg.FERPlusEmotionClassifier())


if __name__ == "__main__":
    unittest.main()
